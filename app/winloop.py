import asyncio


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Custom uvicorn loop factory (pass as `--loop app.winloop:selector_loop_factory`).

    psycopg's async mode requires SelectorEventLoop, but uvicorn's built-in
    "asyncio"/"auto" loop factories unconditionally return ProactorEventLoop on
    win32 - they pass loop_factory= directly to asyncio.run(), which bypasses
    any asyncio.set_event_loop_policy() call entirely. Run uvicorn with
    `--loop app.winloop:selector_loop_factory` (or scripts/serve.py, which
    does this for you) instead of plain `uvicorn app.main:app` on Windows.

    Note: uvicorn calls custom (import-string) loop factories as a plain
    zero-arg callable and uses the return value directly as the event loop -
    unlike its built-in named factories, which are called with
    use_subprocess= and are expected to return *another* callable. Returning
    the SelectorEventLoop class itself here (instead of an instance) causes
    asyncio.Runner to invoke its methods unbound and fail.
    """
    return asyncio.SelectorEventLoop()
