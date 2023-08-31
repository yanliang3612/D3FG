def inflate_batch_array(array, target):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
    return array.view(target_shape)