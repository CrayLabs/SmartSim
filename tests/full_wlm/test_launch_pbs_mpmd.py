import pytest

from smartsim import Experiment
from smartsim.settings import AprunSettings, QsubBatchSettings

pytestmark = pytest.mark.skip(reason="Need to write a compiliation script for hw_mpi.c")


def test_launch_pbs_mpmd():
    """test the launch of a aprun MPMD workload

    this test will obtain an allocation as a batch workload.
    Aprun MPMD workloads share an output file for all processes
    and they share MPI_COMM_WORLDs.

    Prior to running this test, hw_mpi.c in test_configs needs to
    be compiled. #TODO write a script for this.
    """
    exp = Experiment("pbs-test", launcher="pbs")
    run_args = {"pes": 1, "pes-per-node": 1}
    aprun = AprunSettings("./hellow", run_args=run_args)
    aprun2 = AprunSettings("./hellow", run_args=run_args)
    aprun.make_mpmd(aprun2)
    model = exp.create_model("hello_world", run_settings=aprun)

    qsub = QsubBatchSettings(nodes=2, ppn=1, time="1:00:00")
    ensemble = exp.create_ensemble("ensemble", batch_settings=qsub)
    ensemble.add_model(model)

    exp.start(ensemble)
    # TODO write some assertions here.
