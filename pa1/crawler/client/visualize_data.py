from math import log, sqrt
from urllib.parse import urlparse

from pycirclize import Circos
from matplotlib import pyplot as plt

from utils.api_client import APIClient

from pycirclize.utils import ColorCycler
import random

ColorCycler.set_cmap("tab10")

cmap = plt.get_cmap("hsv")  # or "jet", "turbo", "hsv"

def get_rainbow_color(t):
    return cmap(t)  # t ∈ [0, 1]

def visualize_links(page_links):

    sectors = {}
    name2color = {"A": "grey"}

    links_to_add = []

    section_counters = {}
    width = 0.4

    new_id_map = {}

    for link in page_links:
        p1_id = link.get("from_page")
        p2_id = link.get("to_page")
        from_page_url = link.get("from_url")
        to_page_url = link.get("to_url")

        from_parsed = urlparse(from_page_url)
        to_parsed = urlparse(to_page_url)

        from_parsed_section = from_parsed.path.split("/")[1]
        to_parsed_section = to_parsed.path.split("/")[1]

        if ".html" in from_parsed_section:
            from_parsed_section = "other"
        
        if ".html" in to_parsed_section:
            to_parsed_section = "other"

        if from_parsed_section not in section_counters:
            section_counters[from_parsed_section] = 1
        elif p1_id not in new_id_map:
            section_counters[from_parsed_section] += 1

        if p1_id not in new_id_map:
            new_id_map[p1_id] = section_counters[from_parsed_section] - 1

        if to_parsed_section not in section_counters:
            section_counters[to_parsed_section] = 1
        elif p2_id not in new_id_map:
            section_counters[to_parsed_section] += 1

        if p2_id not in new_id_map:
            new_id_map[p2_id] = section_counters[to_parsed_section] - 1

        new_link = ((from_parsed_section, new_id_map[p1_id], new_id_map[p1_id] + width), (to_parsed_section, new_id_map[p2_id], new_id_map[p2_id] + width))
        # print(new_link)
        links_to_add.append(new_link)

    #print(new_id_map)
    # print(section_counters)
    sorted_sectors = dict(
        sorted(section_counters.items(), key=lambda x: x[1], reverse=True)
    )
    sector_to_id = {}
    sector_items = list(sorted_sectors.keys())
    for i in range(len(sorted_sectors)):
        sector_to_id[sector_items[i]] = i
    circos = Circos(sorted_sectors, space=1)
    for i,sector in enumerate(circos.sectors):
        angle = (sector.deg_lim[0] + sector.deg_lim[1]) / 2
        # print(sector)
        track = sector.add_track((92, 100))
        track.axis(fc=ColorCycler(i))
        ha = "right" if angle > 180 else "left"
        if angle > 300:
            rotation = -angle - 90
            track.text(
                sector.name[:20] + ("..." if len(sector.name) > 20 else ""), 
                color="black",
                r=105 + len(sector.name[:20]),
                rotation=rotation,
                adjust_rotation=False,
                size=7 + log(sector.deg_size+1))
        else:
            track.text(
                sector.name[:20] + ("..." if len(sector.name) > 20 else ""), 
                color="black",
                r=105,
                ha=ha,
                adjust_rotation=False,
                size=12)

    for i,l in enumerate(links_to_add):
        #print(l)
        from_sector_id = sector_to_id[l[0][0]]
        circos.link(l[0], l[1], color=ColorCycler(from_sector_id), alpha=pow(5/(5+sorted_sectors[sector_items[from_sector_id]]), 0.5))
    # Plot links with various styles
    

    fig = circos.plotfig(figsize=(12, 12))
    circos.savefig("plot.png", dpi=180)



def visualize_path(page_links):

    section_counters = {}

    new_id_map = {}

    page_links = sorted(page_links, key=lambda x: x.get("accessed_time"))

    ordered_sections = []

    for link in page_links:
        p1_id = link.get("from_page")
        p2_id = link.get("to_page")
        from_page_url = link.get("from_url")
        to_page_url = link.get("to_url")

        from_parsed = urlparse(from_page_url)

        from_parsed_section = from_parsed.path.split("/")[1]

        if ".html" in from_parsed_section:
            from_parsed_section = "other"

        if from_parsed_section not in section_counters:
            section_counters[from_parsed_section] = 1
        else:
            section_counters[from_parsed_section] += 1

        ordered_sections.append(from_parsed_section)

    time_slot_size = 200
    counter = 0
    time_slot_counter = {s: 0 for s in section_counters.keys()}

    stack_plot_data = {s: [] for s in section_counters.keys()}

    for section in ordered_sections:
        if section not in time_slot_counter:
            time_slot_counter[section] = 1
        else:
            time_slot_counter[section] += 1
        counter += 1
        if counter >= time_slot_size:
            for k,v in time_slot_counter.items():
                stack_plot_data[k].append(v)
            counter = 0
            time_slot_counter = {s: 0 for s in section_counters.keys()}

    #print(new_id_map)

    fig, ax = plt.subplots()
    ax.stackplot(range(0, len(links)-time_slot_size, time_slot_size), stack_plot_data.values(),
                labels=stack_plot_data.keys(), alpha=1)
    ax.legend(loc='lower right')
    ax.set_title('Page topics crawled')
    ax.set_xlabel('Time')
    ax.set_ylabel('Num. pages')
    plt.savefig("crawler_path.png")
    


if __name__ == "__main__":

    database_base_url: str = 'https://server:5000'
    _db_api = APIClient(base_url=database_base_url)

    links = _db_api.list_links_with_urls()

    k = 100000
    k = min(k, len(links))
    sampled_links = random.sample(links, k)

    print(sampled_links[0])
    print(len(sampled_links))

    visualize_links(sampled_links)