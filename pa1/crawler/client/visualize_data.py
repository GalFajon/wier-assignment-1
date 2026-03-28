from pycirclize import Circos
from matplotlib import pyplot as plt

from utils.api_client import APIClient

def visualize_links(page_links):

    sectors = {"A": 20000}
    name2color = {"A": "grey"}
    circos = Circos(sectors, space=5)
    for sector in circos.sectors:
        track = sector.add_track((97, 100))
        track.axis(fc=name2color[sector.name])
        track.text(sector.name, color="white", size=12)

    # Plot links with various styles
    width = 0.1
    for link in page_links:
        p1_id = link.get("from_page")
        p2_id = link.get("to_page")
        circos.link(("A", p1_id, p1_id + width), ("A", p2_id, p2_id + width))

    fig = circos.plotfig()
    circos.savefig("plot.png")

if __name__ == "__main__":

    database_base_url: str = 'https://server:5000'
    _db_api = APIClient(base_url=database_base_url)

    links = _db_api.list_links()
    print(len(links))

    visualize_links(links)