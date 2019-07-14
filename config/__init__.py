from .config import *

# init environment

def init_env():
    import os
    import os.path as osp
    os.environ['CUDA_VISIBLE_DEVICES'] = available_gpus

    def check_dir(_dir, create=True):
        if not osp.exists(_dir):
            if create:
                os.makedirs(_dir)
            else:
                raise FileNotFoundError(f'{_dir} not exist')

    check_dir(result_root)
    check_dir(ckpt_folder)

    # check_dir(pv_net.ckpt_record_folder)
    # check_dir(view_net.ckpt_record_folder)
    # check_dir(pc_net.ckpt_record_folder)

    check_dir(view_net.data_root, create=False)
    check_dir(pc_net.data_root, create=False)