import sys
sys.path.append('tests/test_ipu')
import test_ipu_model
import test_ipu_runner
import test_complexdatamanager
import test_hooks
import test_utils

test_ipu_model.test_compare_feat()
test_ipu_model.test_run_model()
test_ipu_model.test_build_model()

test_ipu_runner.test_build_runner()

test_complexdatamanager.test_complexdatamanager()

test_hooks.test_optimizerhook()

test_utils.test_build_from_cfg()
test_utils.test_parse_ipu_options()
