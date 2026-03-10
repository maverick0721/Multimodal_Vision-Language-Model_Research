import webdataset as wds
import torchvision.transforms as T
from PIL import Image


transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])


def preprocess(sample):

    image = sample["jpg"]
    text = sample["txt"]

    image = transform(image)

    tokens = [ord(c) % 32000 for c in text]

    return image, tokens


def create_loader(
    shards,
    batch_size=8,
    num_workers=4
):

    dataset = (
        wds.WebDataset(shards)
        .decode("pil")
        .to_tuple("jpg","txt")
        .map_tuple(transform, lambda x: x)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return loader