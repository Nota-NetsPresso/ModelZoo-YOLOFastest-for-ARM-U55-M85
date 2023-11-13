import os
import argparse
import shutil

import yaml
import torch
import torch.fx as fx
from loguru import logger

from netspresso.client import SessionClient
from netspresso.compressor import ModelCompressor, Task, Framework
from netspresso.launcher import (
    ModelConverter,
    ModelBenchmarker,
    ModelFramework,
    DeviceName,
    BenchmarkTask,
    ConversionTask,
    Model,
    DataType,
    HardwareType,
)

import train
import val
from export import run


def parse_args():
    parser = argparse.ArgumentParser()

    """
        Common arguments
    """
    parser.add_argument('-n', '--name', type=str)

    """
        Compression arguments
    """
    parser.add_argument('-w', '--weight_path', type=str)
    parser.add_argument("--compression_method", type=str, choices=["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"], default="PR_L2")
    parser.add_argument("--recommendation_method", type=str, choices=["slamp", "vbmf"], default="slamp")
    parser.add_argument("--compression_ratio", type=int, default=0.3)
    parser.add_argument("-m", "--np_email", help="NetsPresso login e-mail", type=str)
    parser.add_argument("-p", "--np_password", help="NetsPresso login password", type=str)

    """
        Fine-tuning arguments
    """
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=256, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    """
        Export arguments
    """
    parser.add_argument('--export_half', action='store_true', default=False, help='Entity')

    """
        Converting and benchmark arguments
    """
    parser.add_argument('--target_device', type=str, choices=["Renesas-RA8D1", "Ensemble-E7-DevKit-Gen2"], default="Renesas-RA8D1")
    parser.add_argument('--helium', action='store_true', default=False)

    """
        TFLite file inference arguments
    """
    parser.add_argument('--swap_output_order', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.export_half:
        assert args.device != 'cpu', "Cannot export model to fp16 onnx with cpu mode!!"

    """
        Log in to NetsPresso
    """
    session = SessionClient(email=args.np_email, password=args.np_password)
    compressor = ModelCompressor(user_session=session)
    converter = ModelConverter(user_session=session)
    benchmarker = ModelBenchmarker(user_session=session)

    """ 
        Convert YOLO_Fastest model to fx 
    """
    logger.info("yolo_fastest to fx graph start.")

    model = torch.load(args.weight_path, map_location='cpu')

    #save model_torchfx.pt
    if isinstance(model, torch.nn.Sequential):
        model.train()
        model_head = model[-1]
        model_head.export = False
        model_head.float()
        torch.save(model_head, f'{args.name}_head.pt')
        model = model[0]        
    else:
        model = model['model']
        model.train()
        model_head = model.model[-1]
        model_head.export = False
        model_head.float()
        torch.save(model_head, f'{args.name}_head.pt')
        model.model[-1].exp_yolo_fastest = True
        
        _graph = fx.Tracer().trace(model)
        model = fx.GraphModule(model, _graph)

    model.float()
    torch.save(model, f'{args.name}_fx.pt')
    
    logger.info("yolo_fastest to fx graph end.")

    """
        Model compression - recommendation compression 
    """
    logger.info("Compression step start.")

    UPLOAD_MODEL_NAME = args.name
    TASK = Task.OBJECT_DETECTION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = args.name + '_fx.pt'
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": (args.imgsz, args.imgsz)}]
    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    COMPRESSION_METHOD = args.compression_method
    RECOMMENDATION_METHOD = args.recommendation_method
    RECOMMENDATION_RATIO = args.compression_ratio
    COMPRESSED_MODEL_NAME = f'{UPLOAD_MODEL_NAME}_{COMPRESSION_METHOD}_{RECOMMENDATION_RATIO}'
    OUTPUT_PATH = COMPRESSED_MODEL_NAME + '.pt'
    compressed_model = compressor.recommendation_compression(
        model_id=model.model_id,
        model_name=COMPRESSED_MODEL_NAME,
        compression_method=COMPRESSION_METHOD,
        recommendation_method=RECOMMENDATION_METHOD,
        recommendation_ratio=RECOMMENDATION_RATIO,
        output_path=OUTPUT_PATH,
    )

    logger.info("Compression step end.")

    """
        Retrain YOLO_Fastest model
    """
    logger.info("Fine-tuning step start.")

    backbone_path = OUTPUT_PATH
    backbone = torch.load(backbone_path, map_location='cpu')
    backbone_path = '.'.join(backbone_path.split('.')[:-1]) + '_backbone.pt'
    torch.save(backbone, backbone_path)

    head_path = f'{args.name}_head.pt'
    head = torch.load(head_path, map_location='cpu')
    head_path = '.'.join(OUTPUT_PATH.split('.')[:-1]) + '_head.pt'
    torch.save(head, head_path)

    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)
        hyp['lr0'] *= 0.1
    
    with open('tmp_hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f)
    args.hyp = 'tmp_hyp.yaml'

    run_input = {'netspresso': True, 'weights': backbone_path, **args.__dict__}
    train_opt = train.run(**run_input)

    os.remove(args.hyp)
    logger.info("Fine-tuning step end.")

    """ 
        Export YOLO_Fastest model to onnx
    """
    logger.info("Export model to onnx format step start.")
    
    onnx_save_path = run(
        weights=os.path.join(train_opt.save_dir, 'weights', 'best.pt'),  # weights path
        imgsz=(args.imgsz, args.imgsz),  # image (height, width)
        batch_size=1,  # batch size
        device=args.device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('onnx',),  # include formats
        half=args.export_half,  # FP16 half-precision export
    )

    file_name = onnx_save_path[0].split('/')[-1]
    onnx_model = COMPRESSED_MODEL_NAME + '.onnx'
    shutil.move(onnx_save_path[0], onnx_model)
    
    logger.info(f'=> saving model to {COMPRESSED_MODEL_NAME}.onnx')

    logger.info("Export model to onnx format step end.")

    """
        Convert YOLO_Fastest onnx model to tflite
    """
    logger.info("Converting model to tflite step start.")

    TARGET_DEVICE_NAME = args.target_device
    DATA_TYPE = DataType.INT8

    model: Model = converter.upload_model(onnx_model)
    conversion_task: ConversionTask = converter.convert_model(
        model=model,
        input_shape=model.input_shape,
        target_framework=ModelFramework.TENSORFLOW_LITE,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
        wait_until_done=True,
    )

    logger.info(conversion_task)

    tflite_model = onnx_model.replace('.onnx', '.tflite')
    converter.download_converted_model(conversion_task, dst=tflite_model)

    logger.info("Converting model to tflite step end.")

    """
        Benchmark on target device
    """
    logger.info("Benchmark step start.")

    hardware_type = HardwareType.HELIUM if args.helium else None

    benchmark_model: Model = benchmarker.upload_model(tflite_model)
    benchmark_task: BenchmarkTask = benchmarker.benchmark_model(
        model=benchmark_model,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
        hardware_type=hardware_type,
        wait_until_done=True
    )

    logger.info("Benchmark step end.")
    logger.info(f"model inference latency: {benchmark_task.latency} ms")

    """
        TFLite file inference
    """
    logger.info("TFLite file inference step start.")

    val.run(
        data=args.data,
        weights=tflite_model,
        task='val',
        imgsz=args.imgsz,
        anchors_for_tflite_path=train_opt.save_dir + '/weights/anchors.json',
        swap_output_order=args.swap_output_order,
    )

    logger.info("TFLite file inference step end.")
