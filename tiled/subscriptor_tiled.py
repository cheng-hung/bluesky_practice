from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_kafka.consume import BasicConsumer
from tiled.client import from_profile, from_uri

tiled_sandbox = from_uri('https://tiled.nsls2.bnl.gov')['pdf']['sandbox']

def tiled_subscriptor(tiled_client):

    # factory = plugin0_factory(beamline_acronym, ini_config)
    # router = plugin0_Router(beamline_acronym, ini_config)
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

    rd = RemoteDispatcher(
        "ipc:///var/lib/bluesky-zmq-proxy/pdf-ipc-in-ipc-out/out.sock"
    )
    # install_qt_kicker(rd.loop)
    rd.subscribe(print_message)
    print("\n\n Subscribe to RemoteDispatcher and start the server \n\n")
    rd.start()