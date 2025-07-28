import threading
import time

done = False


def worker(text, id):
    counter = 0
    while True:
        time.sleep(1)
        counter += 1
        # print(f"Worker thread has run {counter} times.")
        print(f"Worker thread {id} says: {text} and has run {counter} times.")


# This is going to start the worker function in a separate thread from the main thread
threading.Thread(target=worker, daemon=True, args=("ABC", 1, )).start()
threading.Thread(target=worker, daemon=True, args=("XYZ", 2, )).start()


input("Press Enter to stop the worker thread...")
done = True
print("Worker thread has been stopped.")
