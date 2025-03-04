from DataIngestor.DataIngestor import DataIngestor
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    data_ingestor = DataIngestor()
    data_ingestor.ingest_pdf("files/1.pdf", "legalaibot-litigation", "test")
