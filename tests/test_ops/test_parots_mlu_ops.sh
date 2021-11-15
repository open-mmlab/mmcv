test_half=$1

pytest test_focal_loss.py::Testfocalloss::test_sigmoid_float -s
pytest test_focal_loss.py::Testfocalloss::test_grad_sigmoid_float -s
pytest test_roi_align.py -s
pytest test_nms.py::Testnms::test_nms_allclose -s


echo "test_half: " $1
if [ $test_half -gt 0 ] ;then
	pytest test_focal_loss.py::Testfocalloss::test_sigmoid_half -s
fi
