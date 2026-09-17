import os
import datetime
import pprint
import uuid
import matplotlib.pyplot as plt
# from bluesky_kafka import RemoteDispatcher
from bluesky_kafka.consume import BasicConsumer

from nslsii.kafka_utils import _read_bluesky_kafka_config_file  # nslsii >=0.7.0



def print_kafka_messages():

    def print_message(consumer, doctype, doc):
        message = doc
        print(
            f"\n{datetime.datetime.now().isoformat()}\n"
            # f"\ndocument keys: {list(message.keys())}\n"
            f"\ncontents: {pprint.pformat(message)}\n"
        )

    # def print_message(consumer, doctype, doc):
    #     name, message = doc
    #     print(
    #         f"\n{datetime.datetime.now().isoformat()} document: {name}\n"
    #     #     f"\ndocument keys: {list(message.keys())}\n"
    #     #     f"\ncontents: {pprint.pformat(message)}\n"
    #     )
    #     if name == 'start':
    #         print(
    #             # f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
    #             f"\ndocument keys: {list(message.keys())}\n"
    #             )
                
    #     elif name == 'event':
    #         print(
    #             # f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
    #             f"\ndocument keys: {list(message.keys())}\n"
    #             )
                
    #     elif name == 'stop':
    #     #     # print('Kafka test good!!')
    #         print(
    #             # f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
    #             f"\ndocument keys: {list(message.keys())}\n"
    #             f"\ncontents: {pprint.pformat(message['num_events'])}\n"
    #             )

    #         print('\n########### Events printing division ############\n')

    kafka_config = _read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")

    # this consumer should not be in a group with other consumers
    #   so generate a unique consumer group id for it
    unique_group_id = f"echo-{'pdf'}-{str(uuid.uuid4())[:8]}"

    kafka_consumer = BasicConsumer(
        topics=['pdf.bluesky.runengine.documents'],
        bootstrap_servers=kafka_config["bootstrap_servers"],
        group_id=unique_group_id,
        consumer_config=kafka_config["runengine_producer_config"],
        process_message = print_message,
    )

    try:
        kafka_consumer.start_polling(work_during_wait=lambda : plt.pause(.1))
    except KeyboardInterrupt:
        print('\nExiting Kafka consumer')
        return()


if __name__ == "__main__":
    import sys
    print_kafka_messages()