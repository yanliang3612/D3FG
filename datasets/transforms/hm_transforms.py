from torchvision.transforms import Compose
import copy
import torch 
from ..protein.constants import * 
from ..molecule.constants import *
from .transform import *

linker_range = torch.tensor([i for i in range(num_atom_types)])
fg_range = torch.tensor([i + num_atom_types for i in range(num_fg_types)])
DEFAULT_PAD_VALUES = {
    'fg_type': num_fg_types + num_aa_types + num_atom_types, 
    'linker_type': num_fg_types + num_aa_types + num_atom_types, 
}
FRAGMENT_DICT = {'aa': 1, 'mol': 2}


@register_transform('mask_ligand')
class MaskLigand():
    def __init__(self):
        super().__init__()
        self.mask_gen_func = self._mask_ligand_for_gen

    def _mask_ligand_for_gen(self, pl_pair):
        if pl_pair['protein'] is not None:
            pl_pair['protein']['ligand_flag'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False,
            )

        if pl_pair['ligand'] is not None:
            pl_pair['ligand']['ligand_flag'] = torch.full_like(
                pl_pair['ligand']['type_fg'],
                fill_value = True,
            )
        return pl_pair

    def __call__(self, pl_pair):
        if pl_pair['protein'] is not None:
            pl_pair['protein']['generate_flag'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False,
            )

        if pl_pair['ligand'] is not None:
            pl_pair['ligand']['generate_flag'] = torch.full_like(
                pl_pair['ligand']['type_fg'],
                fill_value = True,
            )
        
        pl_pair = self.mask_gen_func(pl_pair)

        return pl_pair


@register_transform('mask_linker_elab')
class MaskLigandLinker(MaskLigand):
    def __init__(self):
        super().__init__()
        self.mask_gen_func = self._mask_linker_for_gen

    def _mask_linker_for_gen(self, pl_pair):
        if pl_pair['protein'] is not None:
            pl_pair['protein']['linker_flag'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False,
            )

        if pl_pair['ligand'] is not None:
            pl_pair['ligand']['linker_flag'] = (
                pl_pair['ligand']['mask_linker']
            ).bool()

        return pl_pair

    def __call__(self, pl_pair):
        pl_pair['ligand']['gen_linker'] = True
        
        pl_pair = self.mask_gen_func(pl_pair)

        return pl_pair

@register_transform('mask_cold_fg')
class MaskHotFG(MaskLigand):
    def __init__(self):
        super().__init__()
        self.mask_gen_func = self._mask_cold_fg_for_gen

    def _mask_cold_fg_for_gen(self, pl_pair):
        if pl_pair['protein'] is not None:
            pl_pair['protein']['fg_flag'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False,
            )

        if pl_pair['ligand'] is not None:
            pl_pair['ligand']['fg_flag'] = torch.full_like(
                pl_pair['ligand']['type_fg'],
                fill_value = False
            )

            coldest_idx = pl_pair['ligand']['score_fg'].argmin()
            pl_pair['ligand']['fg_flag'][coldest_idx] = True
            pl_pair['ligand']['fg_flag'][coldest_idx] = True
            pl_pair['ligand']['fg_idx'] = str(',').join(str(idx) for idx in pl_pair['ligand']['fg_idx'][coldest_idx])
        
        return pl_pair
        

    def __call__(self, pl_pair):

        pl_pair['ligand']['gen_linker'] = False
        
        pl_pair = self.mask_gen_func(pl_pair)
        
        return pl_pair

@register_transform('mask_hot_fg')
class MaskHotFG(MaskLigand):
    def __init__(self):
        super().__init__()
        self.mask_gen_func = self._mask_hot_fg_for_gen

    def _mask_hot_fg_for_gen(self, pl_pair):
        if pl_pair['protein'] is not None:
            pl_pair['protein']['fg_flag'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False,
            )

        if pl_pair['ligand'] is not None:
            pl_pair['ligand']['fg_flag'] = torch.full_like(
                pl_pair['ligand']['type_fg'],
                fill_value = False
            )

            hotest_idx = pl_pair['ligand']['score_fg'].argmax()
            pl_pair['ligand']['fg_flag'][hotest_idx] = True
            pl_pair['ligand']['fg_idx'] = str(',').join(str(idx) for idx in pl_pair['ligand']['fg_idx'][hotest_idx])
        
        return pl_pair
        

    def __call__(self, pl_pair):
        pl_pair['ligand']['gen_linker'] = False
        
        pl_pair = self.mask_gen_func(pl_pair)

        return pl_pair
        
@register_transform('center_linker')
class CenterGen(object):

    def __init__(self):
        super().__init__()
            
    def __call__(self, concat_data):
        
        if 'gen_com' in concat_data:
            fg_pos_com = concat_data['gen_com']
        else: 
            fg_pos_com = concat_data['cond_com'] 
        
        concat_data['pos_heavyatom'] = (
            concat_data['pos_heavyatom'] 
            - fg_pos_com.expand_as(concat_data['pos_heavyatom'])
            )
        concat_data['pos_heavyatom'] = (
            concat_data['pos_heavyatom'] 
            * concat_data['mask_heavyatom'].unsqueeze(dim=-1)
        )
        
        # concat_data['pos_heavyatom'][concat_data['fg_flag']][concat_data['mask_heavyatom'][concat_data['fg_flag']]].mean(dim=0)
        return concat_data
        
    
@register_transform('merge_pl_hme')
class MergePLHME(object):

    def __init__(self):
        super().__init__()
    
    def merge_linker_as_fg(self, pl_pair_ligand):
        linker_size = pl_pair_ligand['linker_flag'].sum()
        fg_size = pl_pair_ligand['fg_flag'].sum()

        pl_pair_ligand['linker_type'] = pl_pair_ligand['linker_type'][pl_pair_ligand['linker_flag']]

        pl_pair_ligand['linker_pos'] = pl_pair_ligand['pos_linker'][pl_pair_ligand['linker_flag']]

        pl_pair_ligand['fg_type'] = torch.cat([
            pl_pair_ligand['fg_type'], pl_pair_ligand['linker_type']
            ])
        
        pl_pair_ligand['seq_type'] = torch.cat([
            pl_pair_ligand['seq_type'], 
            torch.full_like(
                pl_pair_ligand['linker_type'],
                fill_value = FRAGMENT_DICT['mol'],
                )
            ])

        pl_pair_ligand['res_nb'] = torch.cat([
            pl_pair_ligand['res_nb'], 
            torch.zeros_like(
                pl_pair_ligand['linker_type']
                )
            ])

        pl_pair_ligand['ligand_flag'] = torch.cat([
            pl_pair_ligand['ligand_flag'], 
            torch.ones_like(
                pl_pair_ligand['linker_type']
                )
            ])

        linker_pos_heavyatom = torch.zeros((linker_size, 15, 3))
        linker_pos_heavyatom[:, BBHeavyAtom.CA] = pl_pair_ligand['linker_pos']
        pl_pair_ligand['pos_heavyatom'] = torch.cat(
            [pl_pair_ligand['pos_heavyatom'], linker_pos_heavyatom], 
            dim=0
            )
        
        linker_mask_heavyatom = torch.zeros((linker_size, 15)).bool()
        linker_mask_heavyatom[:, BBHeavyAtom.CA] = True
        pl_pair_ligand['mask_heavyatom'] = torch.cat(
            [pl_pair_ligand['mask_heavyatom'], linker_mask_heavyatom], 
            dim=0
            )

        pl_pair_ligand['fg_flag'] = torch.cat(
            [
            pl_pair_ligand['fg_flag'], 
            torch.full_like(
            pl_pair_ligand['linker_type'],
            fill_value=pl_pair_ligand['gen_linker']
            )
            ]
        ).bool()

        assert(pl_pair_ligand['fg_flag'].shape[0] == pl_pair_ligand['mask_heavyatom'].shape[0] == 
               pl_pair_ligand['pos_heavyatom'].shape[0] == pl_pair_ligand['res_nb'].shape[0] == 
               pl_pair_ligand['seq_type'].shape[0] == pl_pair_ligand['fg_type'].shape[0])

        return pl_pair_ligand
            
    def __call__(self, pl_pair):
        
        data_list = []
        if pl_pair['protein'] is not None:
            pl_pair['protein']['seq_type'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = FRAGMENT_DICT['aa'],
            )
            pl_pair['protein']['fg_type'] = pl_pair['protein']['aa']
            pl_pair['protein']['ligand_flag']= torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False
            )
            data_list.append(pl_pair['protein'])

        if pl_pair['ligand'] is not None:
            pl_pair['ligand']['seq_type'] = torch.full_like(
                pl_pair['ligand']['type_fg'],
                fill_value = FRAGMENT_DICT['mol'],
            )
            pl_pair['ligand']['res_nb'] = torch.zeros_like(pl_pair['ligand']['type_fg'])
            shift_types = num_aa_types
            
            pl_pair['ligand']['fg_type'] = pl_pair['ligand']['type_fg'].clone() + shift_types
            
            pl_pair['ligand']['linker_type'] = pl_pair['ligand']['type_linker'].clone() + shift_types
            pl_pair['ligand']['ligand_flag']= torch.full_like(
                pl_pair['ligand']['type_fg'],
                fill_value = True
            )
            pl_pair['ligand'] = self.merge_linker_as_fg(pl_pair['ligand'])

            data_list.append(pl_pair['ligand'])

        tensor_props = {
                'seq_type':[],
                'res_nb':[],
                'fg_type':[],
                'linker_type':[],
                'pos_heavyatom':[],
                'mask_heavyatom':[],
                'linker_flag':[],
                'fg_flag':[],
                'ligand_flag':[]
            }

        for data in data_list:
            for k in tensor_props.keys():
                if k in data.keys():
                    tensor_props[k].append(data[k])
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}

        data_out = {
            'entry':pl_pair['entry'],
            'fg_idx': pl_pair['ligand']['fg_idx'],
            **tensor_props,
        }

        generate_fg_pos = data_out['pos_heavyatom'][data_out['fg_flag'].bool()][:,BBHeavyAtom.CA]
        generate_fg_pos_com = generate_fg_pos.mean(dim=0)

        condition_fg_pos = data_out['pos_heavyatom'][~data_out['fg_flag'].bool()][:,BBHeavyAtom.CA]
        condition_fg_pos_com = condition_fg_pos.mean(dim=0)
        
        # data_out['gen_com'] = generate_fg_pos_com
        data_out['cond_com'] = condition_fg_pos_com
        data_out['cond_size'] = (1 - data_out['fg_flag']).sum().item()
        data_out['fg_size'] = data_out['fg_flag'].sum().item()
        data_out['linker_size'] = data_out['linker_flag'].sum().item()
        data_out['linker_pos'] = pl_pair['ligand']['pos_linker']
        data_out['fg_context_flag'] = torch.full_like(data_out['fg_flag'], fill_value=True)
        data_out['linker_context_flag'] = torch.full_like(data_out['linker_flag'], fill_value=True)

        generate_linker_pos_com = data_out['linker_pos'][data_out['linker_flag']].mean(dim=0)
        data_out['linker_com'] = generate_linker_pos_com

        return data_out



@register_transform('merge_pl_atom')
class MergePLLinker(object):

    def __init__(self):
        super().__init__()
    
    def extend_linker_as_aa(self, pl_pair_ligand):
        linker_size = pl_pair_ligand['linker_flag'].shape[0]

        linker_pos_heavyatom = torch.zeros((linker_size, 15, 3))
        linker_pos_heavyatom[:, BBHeavyAtom.CA] = pl_pair_ligand['pos_linker']
        pl_pair_ligand['pos_heavyatom'] = linker_pos_heavyatom
        
        linker_mask_heavyatom = torch.zeros((linker_size, 15)).bool()
        linker_mask_heavyatom[:, BBHeavyAtom.CA] = True
        pl_pair_ligand['mask_heavyatom'] = linker_mask_heavyatom

        pl_pair_ligand['fg_flag'] = pl_pair_ligand['linker_flag']

        assert(pl_pair_ligand['mask_heavyatom'].shape[0] == 
               pl_pair_ligand['pos_heavyatom'].shape[0] == 
               pl_pair_ligand['res_nb'].shape[0] == 
               pl_pair_ligand['seq_type'].shape[0] )

        return pl_pair_ligand
            
    def __call__(self, pl_pair):
        
        data_list = []
        if pl_pair['protein'] is not None:
            pl_pair['protein']['seq_type'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = FRAGMENT_DICT['aa'],
            )
            pl_pair['protein']['fg_type'] = pl_pair['protein']['aa']
            pl_pair['protein']['fg_flag'] = torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False
            )
            pl_pair['protein']['linker_type'] = pl_pair['protein']['aa']
            pl_pair['protein']['ligand_flag']= torch.full_like(
                pl_pair['protein']['aa'],
                fill_value = False
            )
            data_list.append(pl_pair['protein'])

        if pl_pair['ligand'] is not None:
            pl_pair['ligand']['seq_type'] = torch.full_like(
                pl_pair['ligand']['type_linker'],
                fill_value = FRAGMENT_DICT['mol'],
            )
            pl_pair['ligand']['res_nb'] = torch.zeros_like(pl_pair['ligand']['type_linker'])
            shift_types = num_aa_types
                        
            pl_pair['ligand']['fg_type'] = pl_pair['ligand']['type_linker'].clone() + shift_types
            pl_pair['ligand']['linker_type'] = pl_pair['ligand']['type_linker'].clone() + shift_types
            pl_pair['ligand']['ligand_flag']= torch.full_like(
                pl_pair['ligand']['fg_type'],
                fill_value = True
            )
            pl_pair['ligand'] = self.extend_linker_as_aa(pl_pair['ligand'])

            data_list.append(pl_pair['ligand'])

        tensor_props = {
                'seq_type':[],
                'res_nb':[],
                'fg_type':[],
                'pos_heavyatom':[],
                'mask_heavyatom':[],
                'linker_flag':[],
                'fg_flag':[],
                'linker_type':[],
                'ligand_flag':[]
            }

        for data in data_list:
            for k in tensor_props.keys():
                if k in data.keys():
                    tensor_props[k].append(data[k])
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}

        data_out = {
            'entry':pl_pair['entry'],
            **tensor_props,
        }

        generate_fg_pos = data_out['pos_heavyatom'][data_out['fg_flag'].bool()][:,BBHeavyAtom.CA]
        generate_fg_pos_com = generate_fg_pos.mean(dim=0)

        condition_fg_pos = data_out['pos_heavyatom'][~data_out['fg_flag'].bool()][:,BBHeavyAtom.CA]
        condition_fg_pos_com = condition_fg_pos.mean(dim=0)
        
        data_out['gen_com'] = generate_fg_pos_com
        data_out['cond_com'] = condition_fg_pos_com
        data_out['cond_size'] = (1 - data_out['linker_flag']).sum().item()
        data_out['linker_size'] = data_out['linker_flag'].sum().item()
        data_out['linker_pos'] = pl_pair['ligand']['pos_linker']
        data_out['linker_context_flag'] = torch.full_like(data_out['linker_flag'], fill_value=True)

        generate_linker_pos_com = data_out['linker_pos'][data_out['linker_flag']].mean(dim=0)
        data_out['linker_com'] = generate_linker_pos_com

        return data_out