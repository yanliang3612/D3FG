
import copy
import os
from functools import partial
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import accelerate
import torch
from tqdm import tqdm
import shutil
from .misc import *

from .train_lib import *
from .refine import *
from .mol_builder import * 
from .mol_elaborator import *
class Runner:
    def __init__(
        self,
        data,
        model,
        config,
        config_path,
        epoch_num=0,
    ):
        self.config = config
        self.model = model
        self.data_loaders = data
        ddp_kwargs = accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=True
        )
        self.accelerator = accelerate.Accelerator(
            kwargs_handlers=[ddp_kwargs], mixed_precision=self.config.mixed_precision
        )
        self.optimizer = get_optimizer(self.config.train.optimizer, self.model)
        self.scheduler = get_scheduler(self.config.train.scheduler, self.optimizer)

        world_size = torch.cuda.device_count()
        print("Let's use", world_size, "GPUs!")
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
            )
        self.data_loaders["train"], self.data_loaders["val"] = self.accelerator.prepare(
            self.data_loaders["train"], self.data_loaders["val"]
            )

        if hasattr(self.config.train, "max_grad_norm"):
            self.gradnorm_clip_queue = Queue()
            self.gradnorm_clip_queue.add(self.config.train.max_grad_norm)
            self.clip_gradnorm = partial(
                gradient_clipping,
                gradnorm_queue=self.gradnorm_clip_queue,
            )

        if hasattr(self.config.train, "ema_decay"):
            self.model_ema = copy.deepcopy(model)
            self.ema = EMA(self.config.train.ema_decay)

        self.setup_log(config_path)

        seed_all(self.config.train.seed)

        self.epoch_num = epoch_num

        if epoch_num > 0:
            self.load_model(epoch_num)

    def setup_log(self, config_path):
        self.log_dir = get_log_dir(self.config.log_dir, exp_name=self.config.exp_name)
        self.logger = get_logger("train", self.log_dir, "info.log")
        self.writer = SummaryWriter(self.config.log_dir)
        self.ckpt_path = os.path.join(self.log_dir, "saved_models")
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.logger.info("The current workplace is: {}".format(self.log_dir))
        self.logger.info(self.config)

        if not os.path.exists(os.path.join(self.log_dir, os.path.basename(config_path))):
            shutil.copyfile(
                config_path, os.path.join(self.log_dir, os.path.basename(config_path))
            )

    def log_loss_dict(self, out, epoch, tag, others={}):
        logstr = '[%s] Epoch %05d' % (tag, epoch)
        logstr += ' | loss %.4f' % out['overall'].item()
        for k, v in out.items():
            if k == 'overall': continue
            logstr += ' | loss(%s) %.4f' % (k, v.item())
        for k, v in others.items():
            logstr += ' | %s %2.4f' % (k, v)
        self.logger.info(logstr)

        for k, v in out.items():
            if k == 'overall':
                self.writer.add_scalar('%s/loss' % tag, v, epoch)
            else:
                self.writer.add_scalar('%s/loss_%s' % (tag, k), v, epoch)
        for k, v in others.items():
            self.writer.add_scalar('%s/%s' % (tag, k), v, epoch)
        self.writer.flush()
    
    def save_model(self, epoch_num):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path, exist_ok=True)

        model = self.accelerator.unwrap_model(self.model)
        ckpt = {}
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["epoch_num"] = epoch_num
        ckpt["optimizer"] = self.optimizer.state_dict()
        ckpt["scheduler"] = self.scheduler.state_dict()
        if hasattr(self, "model_ema"):
            ckpt["model_ema_state_dict"] = self.model_ema.state_dict()

        model_name = Path(self.ckpt_path) / ("epo%d.tar" % epoch_num)
        torch.save(ckpt, model_name)
        self.logger.info("Saved model at {}".format(epoch_num))
        return self.model

        
    def load_model(self, epoch_num):
        model_name = Path(self.ckpt_path) / ("epo%d.tar" % epoch_num)

        assert os.path.exists(model_name), "Weights at epoch %d not found" % epoch_num

        ckpt = torch.load(model_name, map_location="cpu")

        self.model = self.accelerator.unwrap_model(self.model)

        self.model.load_state_dict(ckpt["model_state_dict"])

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        if hasattr(self, "model_ema") and ("model_ema_state_dict" in ckpt):
            self.model_ema.load_state_dict(ckpt["model_ema_state_dict"])

        self.logger.info("Loaded model at {}".format(epoch_num))

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
            )

        return self.model

    def train_batch(self, batch, epoch, verbose=False):
        time_start = current_milli_time()

        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_dict = self.model(batch)
        
        time_forward_end = current_milli_time()

        self.accelerator.backward(loss)
        orig_grad_norm = self.clip_gradnorm(self.model, verbose=verbose)
        self.optimizer.step()
        
        time_backward_end = current_milli_time()

        model = self.accelerator.unwrap_model(self.model)
        if hasattr(self, "ema"):
            self.ema.update_model_average(self.model_ema, model)

        if verbose:
            self.log_loss_dict(loss_dict, epoch, 'train', others={
            'grad': orig_grad_norm,
            'lr': self.optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })

        return loss.item()

    @torch.no_grad()
    def validate(self, epoch, model, dataset="val"):
        val_loader = self.data_loaders["{}".format(dataset)]
        
        loss_tape = ValidationLossTape(self.logger, self.writer)

        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
            loss, loss_dict = model(batch)
            loss_tape.update(loss_dict, 1)
        
        avg_loss = loss_tape.log(epoch, 'val')
        
        if self.config.train.scheduler.type == "plateau":
            self.scheduler.step(avg_loss)
        else:
            self.scheduler.step()

        return avg_loss.item()

    def train(self, verbose=False):
        train_loader = self.data_loaders["train"]
        # self.validate(0, self.model, 'val')
        for epoch in range(self.epoch_num, self.config.train.max_epoch):

            if verbose:
                progress_bar = train_loader
            else:
                progress_bar = tqdm(
                        train_loader, 
                        unit="batch", 
                        dynamic_ncols=True, 
                        desc='Training the {} epoch'.format(epoch)
                        )

            for batch in progress_bar:
                loss = self.train_batch(batch, epoch, verbose=verbose)

                if not verbose:
                    progress_bar.set_postfix(training_loss=loss, epoch=epoch)

            if (
                epoch % self.config.train.val_freq == 0
                or epoch == self.config.train.max_epoch
            ):
                avg_val_loss = self.validate(epoch, model=self.model)
                self.save_model(epoch)

                if hasattr(self, "_model_ema"):
                    self.logger.info("Validating ema model...")
                    avg_val_loss_ema = self.validate(epoch, model=self.model_ema)
                    avg_val_loss = avg_val_loss_ema

        return avg_val_loss
    
    @torch.no_grad()
    def generate(self, data_loader=None, sample_num=1000, pregiven_context={}, pregen_model=None, clean_frag=True):

        if data_loader is None and "test" in self.data_loaders:
            data_loader = self.data_loaders["test"]
        else:
            assert data_loader is not None
        
        data_loader = self.accelerator.prepare(data_loader)
        
        if "pdb_path" in pregiven_context.keys():
            self.generate_given_id(data_loader, pregiven_context, sample_num, pregen_model=pregen_model, clean_frag=clean_frag)
        
        else:
            self.generate_recurrsively(data_loader, sample_num, pregen_model=pregen_model, clean_frag=clean_frag)

    @torch.no_grad()
    def generate_recurrsively(self, data_loader, sample_num=100, max_sample_num=1000, pregen_model=None, clean_frag=True):
        progress_bar = tqdm(data_loader, unit="batch")

        sdf_dir = self.log_dir + "/saved_samples"
        if not os.path.exists(sdf_dir):
            os.makedirs(sdf_dir, exist_ok=True)

        dict_generated_pos = {}
        dict_generated_v = {}
        dict_generated_fg = {}
        dict_generated_mol = {}
        dict_generated_protein_num = {}

        dict_valid_mol = {}

        generated_num = 0
        valid_num = 0

        for j in range(max_sample_num):
            for i, batch in enumerate(progress_bar):
                protein_ids = batch['id']
                receptor_names = batch['entry'][1]

                print("protein_id: {}".format(protein_ids.tolist()))
                (
                    batch_cat,
                    (final_pos, final_s),
                    traj
                ) = self.model_ema.sample(batch, pregen_model=pregen_model)

                is_aromatic = batch_cat.get('is_aromatic', None)

                for k, protein_id in enumerate(protein_ids):
                    protein_id = protein_id.item()
                    if protein_id not in dict_generated_pos:
                        dict_generated_pos[protein_id] = []
                        dict_generated_fg[protein_id] = []
                        dict_generated_v[protein_id] = []
                        dict_generated_mol[protein_id] = []
                        dict_valid_mol[protein_id] = []
                        dict_generated_protein_num[protein_id] = 0

                    if dict_generated_protein_num[protein_id] >= sample_num:
                        continue
                    mask_gen = batch_cat['mol_mask'][k]

                    current_pos = (
                        final_pos[k][mask_gen].detach().cpu()
                    )
                    current_fg = (
                        final_s[k][mask_gen].detach().cpu()
                    )
                    
                    current_aromatic = (
                        is_aromatic[k][mask_gen].detach().cpu() if is_aromatic is not None else None
                    )

                    dict_generated_pos[protein_id].append(current_pos)
                    dict_generated_fg[protein_id].append(current_fg)

                    try:
                        rd_mol, smiles = build_mol(
                            current_pos, current_fg, is_aromatic=current_aromatic, clean=clean_frag
                        )

                        if "." not in smiles:
                            message = ("valid mol has been generated for {} : {}".format(
                                    protein_id, smiles
                                    ))
                            self.logger.info(message)
                            dict_valid_mol[protein_id].append(rd_mol)
                            dict_generated_protein_num[protein_id] += 1

                            valid_num += 1
                            saved_path = save_valid_mol(
                                                        sdf_dir, 
                                                        receptor_names[k], 
                                                        dict_generated_protein_num[protein_id],
                                                        rd_mol
                                                        )
                            message = ("saved to {}".format(saved_path))
                            self.logger.info(message)
                            

                        dict_generated_mol[protein_id].append(rd_mol)

                        generated_num += 1
                        
                    except:
                        pass


        valid_ratio = valid_num / generated_num
        torch.save(dict_valid_mol, self.log_dir + "/valid_mol_dict.pt")
        print(
            "generate {} mols in total, {} of them is valid.".format(
                generated_num, valid_ratio * 100
            )
        )
    
    @torch.no_grad()
    def elaborate(self, data_loader=None, sample_num=1000, pregiven_context={}):
        if data_loader is None and "test" in self.data_loaders:
            data_loader = self.data_loaders["test"]
        else:
            assert data_loader is not None
        
        data_loader = self.accelerator.prepare(data_loader)

        if "pdb_path" in pregiven_context.keys():
            self.elaborate_given_id(data_loader, pregiven_context, sample_num)
        
        else:
            self.elaborate_recurrsively(data_loader, sample_num)

    @torch.no_grad()
    def elaborate_recurrsively(self, data_loader, sample_num):
        progress_bar = tqdm(data_loader, unit="batch")

        sdf_dir = self.log_dir + "/saved_samples"
        if not os.path.exists(sdf_dir):
            os.makedirs(sdf_dir, exist_ok=True)

        dict_generated_pos = {}
        dict_generated_v = {}
        dict_generated_fg = {}
        dict_generated_mol = {}
        dict_generated_protein_num = {}

        dict_valid_mol = {}

        generated_num = 0
        valid_num = 0

        for j in range(sample_num):
            for i, batch in enumerate(progress_bar):
                protein_ids = batch['id']
                sdf_names = batch['entry'][1][i]
                fg_idxes = [[int(atom_idx) for atom_idx in fg_idx.split(',')] for fg_idx in batch['fg_idx']]

                print("protein_id: {}".format(protein_ids.tolist()))
                (
                    batch_cat,
                    (final_v, final_pos, final_s),
                    traj
                ) = self.model_ema.elaborate(batch)

                for k, protein_id in enumerate(protein_ids):
                    protein_id = protein_id.item()
                    if protein_id not in dict_generated_pos:
                        dict_generated_pos[protein_id] = []
                        dict_generated_fg[protein_id] = []
                        dict_generated_v[protein_id] = []
                        dict_generated_mol[protein_id] = []
                        dict_valid_mol[protein_id] = []
                        dict_generated_protein_num[protein_id] = 0

                    if dict_generated_protein_num[protein_id] >= sample_num:
                        continue
                    mask_gen = batch_cat['gen_mask'][k]

                    current_pos = (
                        final_pos[k][mask_gen].detach().cpu()
                    )
                    current_fg = (
                        final_s[k][mask_gen].detach().cpu()
                    )
                    current_v = (
                        final_v[k][mask_gen].detach().cpu()
                    )
                    current_fg_idx = (
                        fg_idxes[k]
                    )

                    past_fg = (
                        batch_cat['fg_type'][k][mask_gen].detach().cpu()
                    )

                    dict_generated_pos[protein_id].append(current_pos)
                    dict_generated_fg[protein_id].append(current_fg)
                    dict_generated_v[protein_id].append(current_v)



                    sdf_path = Path(self.config.dataset.raw_path) / sdf_names
                    elaborate_mol(sdf_path, current_fg_idx, current_pos, current_fg, current_v)
