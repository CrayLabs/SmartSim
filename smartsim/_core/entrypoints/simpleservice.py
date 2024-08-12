import signal
import typing as t

from smartsim._core.entrypoints.service import Service


class SimpleService(Service):
    """Mock implementation of a service that counts method invocations
    using the base class event hooks."""

    def __init__(
        self,
        log: t.List[str],
        quit_after: int = 0,
        as_service: bool = False,
        cooldown: int = 0,
        loop_delay: int = 0,
    ) -> None:
        super().__init__(as_service, cooldown, loop_delay)
        self._log = log
        self._quit_after = quit_after
        self.num_iterations = 0
        self.num_starts = 0
        self.num_shutdowns = 0
        self.num_cooldowns = 0
        self.num_can_shutdown = 0
        self.num_delays = 0

    def _on_iteration(self) -> None:
        self.num_iterations += 1

    def _on_interrupt(self) -> None:
        self._quit_after = 0

    def _signals(self) -> t.List[int]:
        return [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]

    def _on_signal(self, signo: int, frame: t.Any) -> None:
        super()._on_signal(signo, frame)
        self._quit_after = 0

    def _on_start(self) -> None:
        self._register_handlers()
        self.num_starts += 1

    def _on_shutdown(self) -> None:
        self.num_shutdowns += 1

    def _on_cooldown_elapsed(self) -> None:
        self.num_cooldowns += 1

    def _on_delay(self) -> None:
        self.num_delays += 1

    def _can_shutdown(self) -> bool:
        self.num_can_shutdown += 1
        if self._quit_after == 0:
            return True

        return self.num_iterations >= self._quit_after


# def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
#     print(f"Received signal {signo}")


# def register_signal_handlers(service: Service) -> None:
#     """Register signal handlers prior to execution"""
#     # make sure to register the cleanup before the start
#     # the process so our signaller will be able to stop
#     # the server process.
#     for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]:
#         signal.signal(sig, handle_signal)


if __name__ == "__main__":
    log = []
    service = SimpleService(log, quit_after=100000, as_service=True, loop_delay=1)

    service.execute()
