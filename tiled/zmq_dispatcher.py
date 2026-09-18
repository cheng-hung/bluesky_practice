import datetime
import pprint
import uuid
from bluesky.callbacks.zmq import RemoteDispatcher


def print_zmq_messages(zmq_address: str = None, prefix: bytes = b''):

    if not isinstance(zmq_address, str):
        zmq_address = "ipc:///var/lib/bluesky-zmq-proxy/pdf-ipc-in-ipc-out/out.sock"

    print(f"Listening for Kafka messages for {zmq_address}")

    def print_message(name, doc):
        message = doc
        print(
            f"\n{datetime.datetime.now().isoformat()} document: {name}\n"
            f"\ndocument keys: {list(message.keys())}\n"
            f"\ncontents: {pprint.pformat(message)}\n"
        )

        # fig, ax = plt.subplots()
        # x = np.arange(-10, 10, 0.1)
        # y = np.sin(x)
        # ax.plot(x,y)
        # fig.canvas.manager.show()
        # fig.canvas.flush_events()

    rd = RemoteDispatcher(zmq_address, prefix=prefix)
    # install_qt_kicker(rd.loop)
    rd.subscribe(print_message)
    print("\n\n Subscribe to RemoteDispatcher and start the server \n\n")
    rd.start()



if __name__ == "__main__":
    import sys
    zmq_address = "ipc:///var/lib/bluesky-zmq-proxy/pdf-ipc-in-ipc-out/out.sock"
    prefix=b'reduced'
    print_zmq_messages(zmq_address, prefix=prefix)