from easydict import EasyDict
from detectron2.config import LazyCall as L
from detectron2 import model_zoo
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import ExponentialParamScheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.dataset import DataGenerator
from evaluation import MattingEvaluator
from modeling.SDMatte import SDMatte

hy_dict = EasyDict(
    bsz=1,
    num_workers=8,
    num_GPU=8,
    crop_size=512,
    epoch_num=10,
    data_num=24401,
    losses=["unknown_l1_loss", "known_l1_loss", "known_lap_loss", "unknown_lap_loss"],
    output_dir="output/",
    init_checkpoint=None,
    lr=5e-5,
    resume=False,
    psm="gauss",
    radius=25,
    distillation=False,
    distill_losses=["mse_loss"],
    model_kwargs=EasyDict(
        pretrained_model_name_or_path="LongfeiHuang/SDMatte",
        load_weight=False,
        conv_scale=3,
        num_inference_steps=1,
        aux_input="bbox_mask",
        add_noise=False,
        use_dis_loss=True,
        use_aux_input=True,
        use_coor_input=True,
        use_attention_mask=True,
        residual_connection=False,
        use_encoder_hidden_states=True,
        use_attention_mask_list=[True, True, True],
        use_encoder_hidden_states_list=[False, True, False],
    ),
)

max_iter = int(hy_dict.data_num / hy_dict.bsz / hy_dict.num_GPU * hy_dict.epoch_num)
val_step = int(max_iter / 10)

model = L(SDMatte)(**hy_dict.model_kwargs)

dataloader = OmegaConf.create()
train_dataset = L(DataGenerator)(
    set_list=["Composition-1K", "DIS-646", "AM-2K", "RefMatte", "UHRSD"],
    phase="train",
    crop_size=hy_dict.crop_size,
    psm=hy_dict.psm,
    radius=hy_dict.radius,
)

aim500_test_dataset = L(DataGenerator)(set_list="AIM-500", phase="test", psm=hy_dict.psm, radius=hy_dict.radius)
am2k_test_dataset = L(DataGenerator)(set_list="AM-2K", phase="test", psm=hy_dict.psm, radius=hy_dict.radius)
p3m500_test_dataset = L(DataGenerator)(set_list="P3M-500-NP", phase="test", psm=hy_dict.psm, radius=hy_dict.radius)
rw100_test_dataset = L(DataGenerator)(set_list="RefMatte_RW_100", phase="test", psm=hy_dict.psm, radius=hy_dict.radius)

dataloader.train = L(DataLoader)(
    dataset=train_dataset,
    batch_size=hy_dict.bsz,
    shuffle=False,
    num_workers=hy_dict.num_workers,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset=train_dataset,
        drop_last=True,
    ),
    drop_last=True,
)

dataloader.aim500_test = L(DataLoader)(dataset=aim500_test_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)
dataloader.am2k_test = L(DataLoader)(dataset=am2k_test_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)
dataloader.p3m500_test = L(DataLoader)(dataset=p3m500_test_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)
dataloader.rw100_test = L(DataLoader)(dataset=rw100_test_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

dataloader.evaluator = L(MattingEvaluator)()

train = EasyDict(
    output_dir=hy_dict.output_dir,
    init_checkpoint=hy_dict.init_checkpoint,
    max_iter=max_iter,
    amp=EasyDict(enabled=False),
    ddp=EasyDict(
        broadcast_buffers=True,
        find_unused_parameters=True,
        fp16_compression=True,
    ),
    checkpointer=EasyDict(period=val_step, max_to_keep=100),
    eval_period=val_step,
    log_period=10,
    device="cuda",
)


optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.lr = hy_dict.lr


lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(ExponentialParamScheduler)(
        start_value=1.0,
        decay=5e-3,
    ),
    warmup_length=200 / train.max_iter,
    warmup_factor=0.001,
)
