import uuid
from bluesky_kafka.produce import BasicProducer

basic_producer = BasicProducer(
    topic="some.topic",
    bootstrap_servers=["localhost:9092"],
    key=str(uuid.uuid4())
)

ten_messages = list(range(10))
produced_messages = []
for message in ten_messages:
    basic_producer.produce(message)

basic_producer.flush()