import os
os.environ['PROJECT'] = 'retinanet_test_1'

from edge.detection import get_analytics
from pprint import pprint

analytics_result = get_analytics(
        image_dir = "images/",
        xml_dir = "annotations/pascalvoc_xml/",
        project_path = "/"
)

pprint(analytics_result)



from edge.detection.retinanet import train

train(
    resize_height=600,
    resize_width=600,
    num_epochs=200,
    batch_size=2,
    checkpoint_prefix="try1",
    snapshot_every_epoch=10,
    steps_per_epoch=None,
    sizes=[16,32,64,128, 256],
    strides=[8,16,32,64,128],
    ratios=[0.5,1,2.0],
    scales=[1,1.25,1.5],
    initial_epoch=0,
    weights='imagenet',
    backbone_network='resnet50',
    lr=1e-5,
    print_summary=True)