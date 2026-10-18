import pytest

from conftest import best_match
from vc_label.boxes import iou
from vc_label.contracts import Box
from vc_label.teachers.owlv2 import crop_window

# Frame 006005 of the sg_drive_600s sample: the plate of the red car in the right lane.
SAMPLE_PLATE = Box("plate", 1189, 793, 1245, 808)


def test_crop_window_grows_and_stays_inside_the_image():
    assert crop_window(Box("vehicle", 100, 100, 200, 150), 1000, 1000) == (90, 95, 210, 155)
    assert crop_window(Box("vehicle", 0, 0, 100, 50), 105, 1000) == (0, 0, 105, 55)


@pytest.mark.models
def test_coco_and_owlv2_find_the_sample_car_and_plate(sample_frame, device):
    from vc_label.teachers.coco import CocoTeacher
    from vc_label.teachers.owlv2 import Owlv2PlateTeacher

    aux = CocoTeacher(device).detect(sample_frame)
    car = next(a for a in aux if a.cls == "vehicle" and a.src == "teacher_coco:car" and a.width > 150)
    plates = Owlv2PlateTeacher(device).detect(sample_frame, [car])
    best = best_match(plates, SAMPLE_PLATE)
    assert best is not None and iou(best, SAMPLE_PLATE) >= 0.5 and best.score >= 0.3
