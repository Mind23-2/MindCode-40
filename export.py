import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_param_into_net
from src.config.config import ESRGAN_config,PSNR_config
from src.utils import get_network, resume_model


if __name__ == '__main__':

    config_psnr = PSNR_config
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    model_psnr = RRDBNet(
        in_nc=config_psnr["ch_size"],
        out_nc=config_psnr["ch_size"],
        nf=config_psnr["G_nf"],
        nb=config_psnr["G_nb"],
    )

    model_psnr.set_train(True)
     param_dict_gan = load_checkpoint(args_opt.ganckpt_path)
    param_dict_psnr  = load_checkpoint(args_opt.psnrckpt_path)
    param_dict = OrderedDict()
    alpha = args_opt.alpha
    print('Interpolating with alpha = ', alpha)

    for name,cell_PSNR in net_PSNR.cells_and_names():
        cell_ESRGAN = param_dict_gan[name]
        net_interp[name] = (1 - alpha) * cell_PSNR + alpha * cell_ESRGAN
    load_param_into_net(model_psnr, param_dict)

    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 3, 32, 32)).astype(np.float32))
    input_label = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 3, 128,128)).astype(np.float32))
    G_file = f"ESRGAN_Generator"
    export(G, input_array, file_name=G_file + '-300_11.air', file_format='AIR')