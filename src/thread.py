import threading
import time


def crawl(link, delay=3):
    print(f"crawl started for {link}")
    time.sleep(delay)  # Blocking I/O (simulating a network request)
    print(f"crawl ended for {link}")

links = [
    "https://python.org",
    "https://docs.python.org",
    "https://peps.python.org",
]


threads = []
for link in links:
    t = threading.Thread(target=crawl, args=(link, ), kwargs={"delay": 2})
    threads.append(t)


for thread in threads:
    thread.start()

for t in threads:
    t.join()

