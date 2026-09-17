import uuid
from bluesky_kafka.produce import BasicProducer
from nslsii.kafka_utils import _read_bluesky_kafka_config_file

kafka_config = _read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")

basic_producer = BasicProducer(
    topic="pdf.bluesky.runengine.documents",
    bootstrap_servers=kafka_config["bootstrap_servers"], 
    producer_config=kafka_config["runengine_producer_config"], 
    key=str(uuid.uuid4())
)

ten_messages = list(range(10))
produced_messages = []
for message in ten_messages:
    basic_producer.produce(message)

basic_producer.flush()