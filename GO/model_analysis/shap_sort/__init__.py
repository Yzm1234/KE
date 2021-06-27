import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as colors

class customized_shap:
    def __init__(self):
        """
        This group_dict is flexiable, can be adjusted if nesseccary.
        
        Biomes without grouping:
        
        0 root:Engineered:Biogas plant
        1 root:Engineered:Biogas plant:Wet fermentation
        2 root:Engineered:Bioreactor
        3 root:Engineered:Bioreactor:Continuous culture:Marine sediment inoculum:Wadden Sea-Germany
        4 root:Engineered:Bioremediation:Terephthalate:Wastewater
        5 root:Engineered:Built environment
        6 root:Engineered:Food production
        7 root:Engineered:Food production:Dairy products
        8 root:Engineered:Food production:Fermented beverages
        9 root:Engineered:Food production:Fermented vegetables
        10 root:Engineered:Modeled:Simulated communities (microbial mixture)
        11 root:Engineered:Modeled:Simulated communities (sequence read mixture)
        12 root:Engineered:Solid waste:Composting
        13 root:Engineered:Wastewater
        14 root:Engineered:Wastewater:Activated Sludge
        15 root:Engineered:Wastewater:Industrial wastewater
        16 root:Engineered:Wastewater:Water and sludge
        17 root:Environmental:Aquatic:Aquaculture
        18 root:Environmental:Aquatic:Estuary
        19 root:Environmental:Aquatic:Estuary:Sediment
        20 root:Environmental:Aquatic:Freshwater
        21 root:Environmental:Aquatic:Freshwater:Groundwater
        22 root:Environmental:Aquatic:Freshwater:Groundwater:Mine drainage
        23 root:Environmental:Aquatic:Freshwater:Lake
        24 root:Environmental:Aquatic:Freshwater:Lentic
        25 root:Environmental:Aquatic:Freshwater:Lotic:Sediment
        26 root:Environmental:Aquatic:Freshwater:Pond
        27 root:Environmental:Aquatic:Freshwater:Pond:Sediment
        28 root:Environmental:Aquatic:Freshwater:Sediment
        29 root:Environmental:Aquatic:Lentic:Brackish
        30 root:Environmental:Aquatic:Marine
        31 root:Environmental:Aquatic:Marine:Brackish
        32 root:Environmental:Aquatic:Marine:Coastal
        33 root:Environmental:Aquatic:Marine:Coastal:Sediment
        34 root:Environmental:Aquatic:Marine:Hydrothermal vents
        35 root:Environmental:Aquatic:Marine:Hydrothermal vents:Black smokers
        36 root:Environmental:Aquatic:Marine:Intertidal zone:Coral reef
        37 root:Environmental:Aquatic:Marine:Intertidal zone:Estuary
        38 root:Environmental:Aquatic:Marine:Intertidal zone:Salt marsh
        39 root:Environmental:Aquatic:Marine:Oceanic
        40 root:Environmental:Aquatic:Marine:Oceanic:Aphotic zone
        41 root:Environmental:Aquatic:Marine:Oceanic:Oil-contaminated
        42 root:Environmental:Aquatic:Marine:Oceanic:Photic zone
        43 root:Environmental:Aquatic:Marine:Pelagic
        44 root:Environmental:Aquatic:Marine:Sediment
        45 root:Environmental:Aquatic:Non-marine Saline and Alkaline:Salt crystallizer pond
        46 root:Environmental:Aquatic:Sediment
        47 root:Environmental:Aquatic:Thermal springs:Hot (42-90C)
        48 root:Environmental:Terrestrial:Soil
        49 root:Environmental:Terrestrial:Soil:Agricultural
        50 root:Environmental:Terrestrial:Soil:Contaminated
        51 root:Environmental:Terrestrial:Soil:Sand
        52 root:Host-associated:Algae:Red algae
        53 root:Host-associated:Animal:Digestive system
        54 root:Host-associated:Animal:Digestive system:Fecal
        55 root:Host-associated:Arthropoda:Digestive system:Gut
        56 root:Host-associated:Birds
        57 root:Host-associated:Birds:Digestive system
        58 root:Host-associated:Birds:Digestive system:Ceca
        59 root:Host-associated:Birds:Digestive system:Digestive tube:Cecum
        60 root:Host-associated:Human
        61 root:Host-associated:Human:Circulatory system
        62 root:Host-associated:Human:Digestive system
        63 root:Host-associated:Human:Digestive system:Intestine
        64 root:Host-associated:Human:Digestive system:Large intestine
        65 root:Host-associated:Human:Digestive system:Large intestine:Fecal
        66 root:Host-associated:Human:Digestive system:Large intestine:Sigmoid colon
        67 root:Host-associated:Human:Digestive system:Oral
        68 root:Host-associated:Human:Digestive system:Oral:Saliva
        69 root:Host-associated:Human:Lympathic system:Lymph nodes
        70 root:Host-associated:Human:Reproductive system:Vagina
        71 root:Host-associated:Human:Respiratory system:Pulmonary system:Lung
        72 root:Host-associated:Human:Skin
        73 root:Host-associated:Human:Skin:Naris
        74 root:Host-associated:Insecta:Digestive system
        75 root:Host-associated:Mammals
        76 root:Host-associated:Mammals:Digestive system
        77 root:Host-associated:Mammals:Digestive system:Fecal
        78 root:Host-associated:Mammals:Digestive system:Foregut:Rumen
        79 root:Host-associated:Mammals:Digestive system:Large intestine
        80 root:Host-associated:Mammals:Digestive system:Large intestine:Fecal
        81 root:Host-associated:Mammals:Digestive system:Stomach:Rumen
        82 root:Host-associated:Mammals:Gastrointestinal tract:Intestine:Fecal
        83 root:Host-associated:Mammals:Respiratory system
        84 root:Host-associated:Microbial:Bacteria
        85 root:Host-associated:Plants
        86 root:Host-associated:Plants:Phylloplane
        87 root:Host-associated:Plants:Rhizosphere
        88 root:Host-associated:Porifera
        89 root:Mixed:Sediment:Sediment:Sediment
        """
        self.group_dict = {'root:Engineered:Biogas plant':[0, 1],
                         'root:Engineered:Bioreactor':[2, 3],
                         'root:Engineered:Bioremediation': [4],
                         'root:Engineered:Built environment': [5],
                         'root:Engineered:Food production': list(range(6, 10)),
                         'root:Engineered:Modeled:Simulated communities':[10, 11],
                         'root:Engineered:Solid waste:Composting':[12],
                         'root:Engineered:Wastewater': list(range(13,17)),
                         'root:Environmental:Aquatic:Aquaculture': [17],
                         'root:Environmental:Aquatic:Estuary': [18,19],
                         'root:Environmental:Aquatic:Freshwater': list(range(20,29)),
                         'root:Environmental:Aquatic:Lentic:Brackish':[29],
                         'root:Environmental:Aquatic:Marine': list(range(30,45)),
                         'root:Environmental:Aquatic:Non-marine Saline and Alkaline:Salt crystallizer pond':[45],
                         'root:Environmental:Aquatic:Sediment': [46],
                         'root:Environmental:Aquatic:Thermal springs:Hot (42-90C)':[47],
                         'root:Environmental:Terrestrial:Soil':list(range(48,52)),
                         'root:Host-associated:Algae:Red algae':[52],
                         'root:Host-associated:Animal:Digestive system':[53,54],
                         'root:Host-associated:Arthropoda:Digestive system:Gut':[55],
                         'root:Host-associated:Birds': list(range(56,60)),
                         'root:Host-associated:Human': list(range(60,74)),
                         'root:Host-associated:Insecta:Digestive system':[74],
                         'root:Host-associated:Mammals': list(range(75,84)),
                         'root:Host-associated:Microbial:Bacteria':[84],
                         'root:Host-associated:Plants':[85,86,87],
                         'root:Host-associated:Porifera':[88],
                         'root:Mixed:Sediment:Sediment:Sediment':[89]}
        
        self.color_map = colors.ListedColormap(['SpringGreen','Violet', 'Red', 'DarkBlue', 'Turquoise', 'DarkOrange' \
                            ,'Yellow', 'SteelBlue', 'Aquamarine', 'SlateBlue', 'Teal', 'Plum', 'YellowGreen', 'LightCoral' \
                            , 'Gold', 'ForestGreen', 'HotPink', 'Khaki', 'Indigo', 'CadetBlue', 'Brown', 'Chartreuse', 'DarkOrchid', 'DimGrey', 'DeepSkyBlue'\
                            ,'Fuchsia', 'LightGreen', 'LightSalmon'])

        
    def sort_shap_plot(self, shap_values, features=None,target_feature=None,  feature_names=None, max_display=None, plot_type=None,
                     color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                     color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                     class_inds=None,
                     cmap=None,
                     auto_size_plot=None,
                     use_log_scale=False):

        import matplotlib.pyplot as pl

        multi_class = False
        if isinstance(shap_values, list):
            multi_class = True
            if plot_type is None:
                plot_type = "bar" # default for multi-output explanations
            assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
        else:
            if plot_type is None:
                plot_type = "dot" # default for single output explanations
            assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

        # default color:
        if color is None:
            if plot_type == 'layered_violin':
                color = "coolwarm"
            elif multi_class:
                color = lambda i: colors.red_blue_circle(i/len(shap_values))
            else:
                color = colors.blue_rgb

        idx2cat = None
        # convert from a DataFrame or other types
        if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
            if feature_names is None:
                feature_names = features.columns
            # feature index to category flag
            idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
            features = features.values

        num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

        if features is not None:
            shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                        "provided data matrix."
            if num_features - 1 == features.shape[1]:
                assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                              "constant offset? Of so just pass shap_values[:,:-1]."
            else:
                assert num_features == features.shape[1], shape_msg

        if use_log_scale:
            pl.xscale('symlog')

        if max_display is None:
            max_display = 20

        if sort:
            # order features by the sum of their effect magnitudes
            if multi_class:
                target_feature_idx = class_names.index(target_feature)
                feature_order = np.argsort(np.mean(np.abs(shap_values), axis=1)[target_feature_idx])
            else:
                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
            feature_order = feature_order[-min(max_display, len(feature_order)):]
        else:
            feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

        row_height = 0.4
        if plot_size == "auto":
            pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
        elif type(plot_size) in (list, tuple):
            pl.gcf().set_size_inches(plot_size[0], plot_size[1])
        elif plot_size is not None:
            pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
        pl.axvline(x=0, color="#999999", zorder=-1)


        if multi_class and plot_type == "bar":
            if class_names is None:
                class_names = ["Class "+str(i) for i in range(len(shap_values))]
            feature_inds = feature_order[:max_display]
            y_pos = np.arange(len(feature_inds))
            left_pos = np.zeros(len(feature_inds))

            if class_inds is None:
                class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
            elif class_inds == "original":
                class_inds = range(len(shap_values))
            for i, ind in enumerate(class_inds):
                global_shap_values = np.abs(shap_values[ind]).mean(0)
                pl.barh(
                    y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',
                    color=color(i), label=class_names[ind]
                )
                left_pos += global_shap_values[feature_inds]
            pl.yticks(y_pos, fontsize=13)
            pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
            pl.legend(frameon=False, fontsize=12)


        pl.gca().xaxis.set_ticks_position('bottom')
        pl.gca().yaxis.set_ticks_position('none')
        pl.gca().spines['right'].set_visible(False)
        pl.gca().spines['top'].set_visible(False)
        pl.gca().spines['left'].set_visible(False)
        pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
        pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
        if plot_type != "bar":
            pl.gca().tick_params('y', length=20, width=0.5, which='major')
        pl.gca().tick_params('x', labelsize=11)
        pl.ylim(-1, len(feature_order))
        if plot_type == "bar":
            pl.xlabel('mean(|SHAP value|)(average impact on model output magnitude)', fontsize=13)
        if show:
            pl.show()