from smartsim._core.launcher.cobalt import cobaltParser


def test_parse_step_id():
    output = "JobName      JobId \n" "=====================\n" "smartsim     507975 \n"
    step_id = cobaltParser.parse_cobalt_step_id(output, "smartsim")
    assert step_id == "507975"


def test_parse_step_status():
    output = "JobName      State \n" "=====================\n" "smartsim     running \n"
    step_id = cobaltParser.parse_cobalt_step_status(output, "smartsim")
    assert step_id == "running"


def test_parse_qsub_out():
    output = (
        "Job routed to queue 'debug-flat-quad'.\n"
        "Memory mode set to flat quad for queue debug-flat-quad\n"
        "507998\n"
    )
    step_id = cobaltParser.parse_qsub_out(output)
    assert step_id == "507998"
