## script to plot phylogenetic tree as shown in Figure 4.1.2

from ete3 import TreeStyle, Tree, TextFace, add_face_to_node, NodeStyle, TreeFace

## create sub trees
Megophrys = Tree('(nasuta)Megophrys;', format=1)
Leptobrachium = Tree('(hasseltii)Leptobrachium;', format=1)
Megophridae = 'Megophridae:4'

Ansonia = Tree('Ansonia;', format=1)
Ingerophrynus = Tree('(parvus)Ingerophrynus;', format=1)
Leptophryne = Tree('(borbonica)Leptophryne;', format=1)
Pelophryne = Tree('(signata)Pelophryne;', format=1)
Phrynoidis = Tree('(asper,juxtasper)Phrynoidis;', format=1)
Bufonidae = 'Bufonidae:7'

Microhyla = Tree('(achatina,heymonsi)Microhyla;', format=1)
Microhylidae = 'Microhylidae:1'

Fejervarja = Tree('(limnocharis)Fejervarja;', format=1)
Limnonectes = Tree('(macrodon, paramacrodon, kuhlii,hikidai,blythii,sisikdagu)Limnonectes;', format=1)
Occidozyga = Tree('(sumatrana)Occidozyga;', format=1)
Dicroglassidae = 'Dicroglassidae:1'

Sumaterana = Tree('(montana:1,crassiovis, dabulescens)Sumaterana;', format=1)
Pulchrana = Tree('(picturata,glandulosa,rawa,debussyi)Pulchrana;', format=1)
Odorrana = Tree('(hosii)Odorrana;', format=1)
Amnirana = Tree('(nicobariensis)Amnirana;', format=1)
Chalcorana = Tree('(rufipes,chalconota)Chalcorana;', format=1)
Huia = Tree('(masonii, sumatrana)Huia;', format=1)
Hylarana = Tree('(erythraea)Hylarana;', format=1)
Ranidae = 'Ranidae:1'

Philautus = Tree('Philautus;', format=1)
Polypedates = Tree('(macrotis, otilophus,leucomystax)Polypedates;', format=1)
Rhacophorus = Tree('(poecilonotus,margaritifer,catamitus,cyanopunctatus,prominanus)Rhacophorus;', format=1)
Nyctixalus = Tree('(pictum)Nyctixalus;', format=1)
Rhacophoridae = 'Rhacophoridae:2'

fork1 = Tree('a:5;', format=1)
fork2 = Tree('a:2;', format=1)
fork3 = Tree('a:2;', format=1)
fork4 = Tree('a:5;', format=1)
fork5 = Tree('a:1;', format=1)

Anura = Tree('Anura;', format=1)

## create tree
t = Tree(
    "(({},({},({},({},({},{})a:1)a:5)a:2)a:2)a:5)Anura;".format(Megophridae, Bufonidae, Microhylidae, Dicroglassidae,
                                                                Ranidae, Rhacophoridae),
    format=1)



families = {'Megophridae': '(Litter Frogs)', 'Bufonidae': '(True Toads)', 'Microhylidae': '(Narrow-mouthed Frogs)',
            'Dicroglassidae': '(True Frogs I)', 'Ranidae': '(True Frogs II)',
            'Rhacophoridae': '(Afro-asian Tree Frogs)'}
genera_trees = [[Megophrys, Leptobrachium], [Ansonia, Ingerophrynus, Leptophryne, Pelophryne, Phrynoidis], [Microhyla],
                [Fejervarja, Limnonectes, Occidozyga],
                [Sumaterana, Pulchrana, Odorrana, Amnirana, Chalcorana, Huia, Hylarana],
                [Philautus, Polypedates, Rhacophorus, Nyctixalus]]

genera = [['Megophrys', 'Leptobrachium'], ['Ansonia', 'Ingerophrynus', 'Leptophryne', 'Pelophryne', 'Phrynoidis'],
          ['Microhyla'],
          ['Fejervarja', 'Limnonectes', 'Occidozyga'],
          ['Sumaterana', 'Pulchrana', 'Odorrana', 'Amnirana', 'Chalcorana', 'Huia', 'Hylarana'],
          ['Philautus', 'Polypedates', 'Rhacophorus', 'Nyctixalus']]

dict_genera = {'Megophrys': '(Horned Frogs)', 'Leptobrachium': '(Litter Frogs)',
          'Ansonia': '(Stream Toads)', 'Ingerophrynus': '(Asian Forest Toads)',
           'Leptophryne': '(Indenesian Tree Toads)', 'Pelophryne': '(Dwarf Toads)', 'Phrynoidis': '(River Toads)',
          'Microhyla': '(Narrow-mouthed Frogs)',
          'Fejervarja': '(Terrestrial Frogs)', 'Limnonectes': '(Fanged Frogs)', 'Occidozyga': '(Puddle Frogs)',
          'Sumaterana': '(Cascade Frogs)', 'Pulchrana': '(Asian Ranid Frogs)', 'Odorrana': '(Smelling Frogs)',
           'Amnirana': '(Asian Ranid Frogs)', 'Chalcorana': '(White-lipped Ranid Frogs)', 'Huia': '(Cascade Frogs)',
           'Hylarana': '(White-lipped Frogs)',
          'Philautus': '(Bush Frogs)', 'Polypedates': '(Whipping Frogs)', 'Rhacophorus': '(Parachuting Frogs)',
           'Nyctixalus': '(Indonesian Tree Frogs)'}

dict_species = {'nasuta': '(Bornean Horned Frog', 'hasseltii': '(Hasselt\'s Toad)', 'parvus': '(Lesser Toad)',
           'borbonica': '(Cross Toad)', 'signata': '(Lowland Dwarf Toad)', 'asper': '(Asian Giant Toad)',
           'juxtasper': '(Giant River Toad)', 'achatina': '(Javan Chorus Frog)', 'heymonsi': '(Dark-sided Chorus Frog)',
           'limnocharis': '(Grass Frog)', 'macrodon': '(Fanged River Frog)', 'kuhlii': '(Kuhl\'s Creek Frog)',
           'hikidai': '(Rivulet Frog)', 'blythii': '(Blyth\'s Frog )', 'paramacrodon': '(Lesser Swamp Frog)',
           'sumatrana': '(Sumatran Puddle Frog)', 'montana': '(Mountain Cascade Frog)',
           'dabulescens': '(Gayo Cascade Frog)', 'picturata': '(Spotted Stream Frog)',
           'glandulosa': '(Rough-sided Frog)', 'hosii': '(Poisonous Rock Frog)', 'nicobariensis': '(Cricket Frog)',
           'chaloconota': '(Brown Stream Frog)', 'masonii': '(Javan Torrent Frog)', 'erythraea': '(Green Paddy Frog)',
           'macrotis': '(Dark-eared Tree Frog)', 'otilophus': '(File-eared Tree Frog)',
           'leucomystax': '(Four-lined Tree Frog)', 'poecilonotus': '(Sumatra Flying Frog)',
           'margaritifer': '(Java Flying Frog)', 'cyanopunctatus': '(Blue-spotted Bush Frog)',
           'prominanus': '(Johore Flying Frog)', 'pictum': '(Peter\'s Tree Frog)'}
colors = {'Megophrys': 'olive', 'Leptobrachium': 'gold', 'Ansonia': 'lavender', 'Ingerophrynus': 'navy',
          'Leptophryne': 'maroon', 'Pelophryne': 'pink', 'Phrynoidis': 'magenta', 'Microhyla': 'red',
          'Fejervarja': 'orange', 'Limnonectes': 'cyan', 'Occidozyga': 'deeppink', 'Sumaterana': 'sienna',
          'Pulchrana': 'darkorchid', 'Odorrana': 'purple', 'Amnirana': 'plum', 'Chalcorana': 'lime', 'Huia': 'tomato',
          'Hylarana': 'gray', 'Philautus': 'green', 'Polypedates': 'yellow', 'Rhacophorus': 'teal',
          'Nyctixalus': 'darkkhaki'}

## create node style for connection between tree and subtrees
ns = NodeStyle()
ns["fgcolor"] = "black"
ns["shape"] = "circle"
ns["vt_line_width"] = 1
ns["hz_line_width"] = 1
ns["vt_line_type"] = 0  # 0 solid, 1 dashed, 2 dotted
ns["hz_line_type"] = 1
ns['size'] = 5
## create node style for tree nodes
ns_genera = NodeStyle()
ns["fgcolor"] = "black"
ns["shape"] = "circle"
ns['size'] = 5
ns["vt_line_width"] = 1
ns["hz_line_width"] = 1
ns["vt_line_type"] = 0  # 0 solid, 1 dashed, 2 dotted
ns["hz_line_type"] = 0

ts_genera = TreeStyle()
ts = TreeStyle()

## create node layout for genera and species nodes
def my_layout(node):
    F = TextFace(node.name, tight_text=True, penwidth=5, fsize=12)
    if node.name is not 'a':
        add_face_to_node(F, node, column=0, position="branch-right")
        node.set_style(ns)
        if node.name in families:
            G = TextFace(families[node.name], fstyle='italic')
            add_face_to_node(G, node, column=0, position="branch-right")
            node.set_style(ns)
    if node.name in dict_genera:
        H = TextFace(dict_genera[node.name], fstyle='italic')
        add_face_to_node(H, node, column=0, position="branch-right")
        node.set_style(ns)
    if node.name in dict_species:
        I = TextFace(dict_species[node.name], fstyle='italic')
        add_face_to_node(I, node, column=0, position="branch-right")
        node.set_style(ns)
        if node.name == 'sumatrana':
            dict_species[node.name] = '(Sumatran Torrent Frog)'


    # if node.is_leaf():
    # node.img_style["size"] = 0
    # img = ImgFace('/home/stine/Desktop/{}.jpg'.format(node.name))
    # add_face_to_node(img, node, column=0, aligned=True)
    for n in Anura.iter_search_nodes():
        if n.dist > 1:
            n.img_style = ns


## create colors for each genus
for idx, family in enumerate(families):
    for cidx, genus_tree in enumerate(genera_trees[idx]):
        tf_genera = TreeFace(genus_tree, ts_genera)
        tf_genera.border.width = 2
        genus = genera[idx][cidx]
        color = colors[str(genus)]
        tf_genera.border.color = color
        (t & family).add_face(tf_genera, column=0, position='aligned')

for n in genus_tree.iter_search_nodes():
    if n.dist == 1:
        n.img_style = ns_genera



ts_genera.show_leaf_name = False
ts_genera.show_scale = False
ts_genera.layout_fn = my_layout
ts.branch_vertical_margin = 10

ts.show_leaf_name = False
ts.branch_vertical_margin = 15
ts.layout_fn = my_layout
ts.draw_guiding_lines = True
ts.guiding_lines_type = 1
ts.show_scale = False
ts.allow_face_overlap = False
# ts.mode = "c"
# ts.arc_start = 180 # 0 degrees = 3 o'clock
# ts.arc_span = 270
t.show(tree_style=ts)
t.render("mytree.png", w=183, units="mm", tree_style=ts)
