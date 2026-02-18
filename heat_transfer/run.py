# from single_hole_problem import compare_results as single_hole_run
# single_hole_run.run_bc1()
# single_hole_run.run_bc2()

from multi_hole_problem.scalability_test import run_scalability_test
run_scalability_test()
from multi_hole_problem.time_study import compare_time
compare_time()