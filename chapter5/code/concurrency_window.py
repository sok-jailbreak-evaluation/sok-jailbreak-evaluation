"""Let the long-form review use freed API capacity as the other stages finish."""
import asyncio
import json
import time


class ReviewWindow:
    def __init__(self, initial, maximum, log):
        self.capacity = initial
        self.maximum = maximum
        self.semaphore = asyncio.Semaphore(initial)
        self.retire = 0
        self.log = log
        self.since = time.monotonic()
        self.responses = self.throttled = 0
        self.record(initial, initial)

    async def __aenter__(self):
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, *exc):
        if self.retire:
            self.retire -= 1
        else:
            self.semaphore.release()

    def record(self, old, new):
        with self.log.open("a") as f:
            f.write(json.dumps(dict(unix=time.time(), previous=old, concurrency=new,
                                    responses=self.responses, http429=self.throttled)) + "\n")

    async def observe(self, response):
        self.responses += 1
        self.throttled += int(response.status_code == 429)
        if time.monotonic() - self.since < 45 or self.responses < 24:
            return
        old = self.capacity
        fraction = self.throttled / self.responses
        if fraction >= 0.06:
            self.capacity = max(12, old * 3 // 4)
        elif fraction <= 0.01:
            self.capacity = min(self.maximum, old + 8)
        if self.capacity > old:
            added = self.capacity - old
            cancel = min(added, self.retire)
            self.retire -= cancel
            for _ in range(added - cancel):
                self.semaphore.release()
        else:
            self.retire += old - self.capacity
        self.record(old, self.capacity)
        self.since = time.monotonic()
        self.responses = self.throttled = 0
