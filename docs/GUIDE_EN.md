# IGNIS — The Complete Educational Guide

**Intelligent Geospatial Neural Inference System**
Next-day wildfire spread prediction over Türkiye from satellite Earth Observation data

| | |
|---|---|
| **Paper ID** | IAC-26,B1,IP,107,x110901 |
| **Symposium** | IAF Earth Observation Symposium (B1), Interactive Presentations (IP) |
| **Congress** | 77th International Astronautical Congress, 5–9 October 2026, Antalya, Türkiye |
| **Authors** | Muhammet Erdem Karakoyun (corresponding), Utku Doruk Kaplan, Furkan Bağıç, Mehmet İpek, Ege Kayseri, Özgür Efe Aksoy |
| **Institution** | Antalya Yusuf Ziya Öner Science High School, Antalya, Türkiye |
| **Final manuscript deadline** | 14 September 2026 |
| **Repository** | github.com/WhiteFoxYT/ignis-ai |

---

## Table of contents

- [0. How to use this guide](#0-how-to-use-this-guide)
- [1. Wildfires and the problem](#1-wildfires-and-the-problem)
  - [1.1 What a wildfire actually is](#11-what-a-wildfire-actually-is)
  - [1.2 The fire behaviour triangle: fuel, weather, topography](#12-the-fire-behaviour-triangle-fuel-weather-topography)
  - [1.3 Why fire runs uphill](#13-why-fire-runs-uphill)
  - [1.4 Wind](#14-wind)
  - [1.5 Fire regimes](#15-fire-regimes)
  - [1.6 Why Türkiye burns](#16-why-türkiye-burns)
  - [1.7 The four phases of disaster management](#17-the-four-phases-of-disaster-management)
  - [1.8 Susceptibility is not spread](#18-susceptibility-is-not-spread)
- [2. Remote sensing from zero](#2-remote-sensing-from-zero)
  - [2.1 The electromagnetic spectrum](#21-the-electromagnetic-spectrum)
  - [2.2 What a satellite actually records](#22-what-a-satellite-actually-records)
  - [2.3 Passive and active sensing](#23-passive-and-active-sensing)
  - [2.4 The three resolutions and the trade-off between them](#24-the-three-resolutions-and-the-trade-off-between-them)
  - [2.5 Orbits](#25-orbits)
  - [2.6 MODIS, Terra and Aqua](#26-modis-terra-and-aqua)
  - [2.7 How a satellite detects a fire](#27-how-a-satellite-detects-a-fire)
  - [2.8 The FireMask confidence classes](#28-the-firemask-confidence-classes)
  - [2.9 NDVI](#29-ndvi)
  - [2.10 Land surface temperature](#210-land-surface-temperature)
  - [2.11 Reanalysis: ERA5-Land](#211-reanalysis-era5-land)
  - [2.12 CHIRPS](#212-chirps)
  - [2.13 SRTM, elevation, slope and aspect](#213-srtm-elevation-slope-and-aspect)
- [3. Geospatial fundamentals and projections](#3-geospatial-fundamentals-and-projections)
  - [3.1 Why the Earth cannot be flattened](#31-why-the-earth-cannot-be-flattened)
  - [3.2 Geographic versus projected coordinates](#32-geographic-versus-projected-coordinates)
  - [3.3 UTM and zone 35N](#33-utm-and-zone-35n)
  - [3.4 EPSG codes](#34-epsg-codes)
  - [3.5 Reprojection and resampling](#35-reprojection-and-resampling)
  - [3.6 Co-registration and scale](#36-co-registration-and-scale)
  - [3.7 Google Earth Engine](#37-google-earth-engine)
- [4. Machine learning from zero](#4-machine-learning-from-zero)
- [5. Deep learning and convolutional networks](#5-deep-learning-and-convolutional-networks)
  - [5.1 The artificial neuron](#51-the-artificial-neuron)
  - [5.2 Activation functions](#52-activation-functions)
  - [5.3 Why fully connected layers fail on images](#53-why-fully-connected-layers-fail-on-images)
  - [5.4 The convolution, worked by hand](#54-the-convolution-worked-by-hand)
  - [5.5 Stride, padding and the receptive field](#55-stride-padding-and-the-receptive-field)
  - [5.6 Pooling](#56-pooling)
  - [5.7 Channels](#57-channels)
  - [5.8 Classification versus semantic segmentation](#58-classification-versus-semantic-segmentation)
  - [5.9 The U-Net](#59-the-u-net)
  - [5.10 Counting the parameters](#510-counting-the-parameters)
- [6. Class imbalance and metrics](#6-class-imbalance-and-metrics)
  - [6.1 The confusion matrix](#61-the-confusion-matrix)
  - [6.2 Why accuracy lies](#62-why-accuracy-lies)
  - [6.3 Precision, recall, F1](#63-precision-recall-f1)
  - [6.4 IoU](#64-iou)
  - [6.5 The ROC curve](#65-the-roc-curve)
  - [6.6 The PR curve and AUC-PR](#66-the-pr-curve-and-auc-pr)
  - [6.7 Our own contradiction: ROC-AUC 0.8468 versus AUC-PR 0.0210](#67-our-own-contradiction-roc-auc-08468-versus-auc-pr-00210)
  - [6.8 Threshold selection and calibration](#68-threshold-selection-and-calibration)
  - [6.9 Baselines, and why persistence is mandatory](#69-baselines-and-why-persistence-is-mandatory)
  - [6.10 Techniques for handling imbalance](#610-techniques-for-handling-imbalance)
- [7. The IGNIS data pipeline, line by line](#7-the-ignis-data-pipeline-line-by-line)
  - [7.1 The eight source products](#71-the-eight-source-products)
  - [7.4 Patch extraction](#74-patch-extraction-stratifiedsample-and-neighborhoodtoarray)
  - [7.5 What is new: fire_next2, valid, and the 32×32 crop](#75-what-is-new-fire_next2-valid-and-the-3232-crop)
  - [7.6 Normalisation: the single most important fix](#76-normalisation-the-single-most-important-fix)
  - [7.7 The circular variable problem: aspect](#77-the-circular-variable-problem-aspect)
  - [7.8 The categorical variable problem: land cover](#78-the-categorical-variable-problem-land-cover)
  - [7.9 Direction-aware data augmentation](#79-direction-aware-data-augmentation)
- [8. Training the model and GPUs](#8-training-the-model-and-gpus)
- [9. Reading the results honestly](#9-reading-the-results-honestly)
  - [9.2 Pixel-level results](#92-pixel-level-results)
  - [9.3 Patch-level classification and the 77 % trap](#93-patch-level-classification-and-the-77--trap)
  - [9.4 The persistence comparison](#94-the-persistence-comparison--the-result-that-matters-most)
  - [9.5 The seven diagnosed causes](#95-the-seven-diagnosed-causes)
  - [9.7 Why honesty is a strength](#97-why-honesty-is-a-strength)
- [10. Questions you may be asked, and how to answer](#10-questions-you-may-be-asked-and-how-to-answer)
- [11. Glossary](#11-glossary)
- [12. References and further reading](#12-references-and-further-reading)

---

## 0. How to use this guide

This guide is written for the six students who will stand in front of an international audience in Antalya in October 2026 and defend IGNIS. It assumes **no prior knowledge**. If you have never heard the word "convolution" (Turkish: *evrişim*), this guide starts before that point. If you already know what a U-Net is, you can skip to Section 6, which is where the real scientific content of this project lives.

The guide has three jobs.

1. **Teach the science.** Every concept is explained in three layers: an intuitive analogy, the mathematical definition, and exactly where that concept appears inside IGNIS.
2. **Teach the English.** All of you are native Turkish speakers and you must present in English. Every technical term is given with its Turkish equivalent in parentheses the first time it appears, and Section 11 is a 150-term bilingual glossary. Read the English terms out loud. At IAC you will need to *say* "class imbalance", not translate it in your head.
3. **Prepare you for the jury.** Section 10 contains 33 hard questions with honest answers. Some of these questions will be asked. Prepare for them.

### Which section answers which question

| If you want to know… | Read |
|---|---|
| Why do fires spread the way they do? | Section 1 |
| How does a satellite "see" a fire from 700 km up? | Section 2 |
| What does EPSG:32635 mean? | Section 3 |
| What is training, validation, overfitting? | Section 4 |
| What is a convolution, and what is a U-Net? | Section 5 |
| **Why are our results bad, and what does "bad" even mean?** | **Section 6 and Section 9** |
| What exactly does our code do to the data? | Section 7 |
| Why do we need a GPU, and why AMD? | Section 8 |
| What will the jury ask me? | Section 10 |
| What does this English word mean in Turkish? | Section 11 |

### One warning before you start

This guide is **honest about the current state of IGNIS**. The model, as of the preliminary run documented in the manuscript, does **not work well**. It loses to the simplest possible prediction. Section 9 says this plainly and explains exactly why.

Do not be embarrassed by this. A congress like IAC is not a competition for the highest number. It is a scientific forum. A team of high-school students who can explain, with measured evidence, *why* their model fails and *what specifically they are changing* is doing better science than a team that reports 0.99 accuracy without knowing that 0.997 comes free from predicting nothing. Your honesty is the strongest thing you have. Learn to defend it.

---

## 1. Wildfires and the problem

### 1.1 What a wildfire actually is

A wildfire (Turkish: *orman yangını*) is a self-sustaining chemical reaction — rapid oxidation of plant material — that propagates across a landscape. Three ingredients are needed, and this is called the **fire triangle** (Turkish: *yangın üçgeni*): **fuel** (Turkish: *yakıt*), **oxygen** (Turkish: *oksijen*) and **heat** (Turkish: *ısı*). Remove any one and the fire stops. Water works by removing heat; a firebreak works by removing fuel; a fire blanket works by removing oxygen.

But the fire triangle only explains *combustion*. It does not explain *spread*. For spread you need a different model, and this is the model IGNIS is built on.

Fire spreads because burning fuel radiates heat and produces hot gas that **preheats** the fuel next to it. Preheating drives out moisture and then drives out volatile gases from the plant material — a process called **pyrolysis** (Turkish: *piroliz*). Once those gases reach ignition temperature, the neighbouring fuel catches, and the process repeats. So fire spread is a chain of preheating events. Anything that makes preheating faster or easier makes the fire move faster.

Three families of factors control preheating. Together they are called the **fire behaviour triangle**.

### 1.2 The fire behaviour triangle: fuel, weather, topography

```
                         FIRE BEHAVIOUR
                              ▲
                             / \
                            /   \
                   FUEL    /     \    WEATHER
              (yakıt)     /       \   (hava durumu)
                         /         \
                        /___________\
                          TOPOGRAPHY
                          (topoğrafya)
```

**Fuel (Turkish: *yakıt*).** How much plant material is there, how is it arranged vertically, and how dry is it? A dense, continuous, dry, resin-rich pine forest carries fire far better than a sparse, green, irrigated orchard. The relevant properties are *load* (mass per unit area), *continuity* (are there gaps?), *arrangement* (fine twigs and needles ignite in seconds; a thick trunk takes hours), and above all **fuel moisture** (Turkish: *yakıt nemi*). Fine dead fuels equilibrate with the air in about an hour, which is why relative humidity is such a powerful predictor.

**In IGNIS**, fuel is represented by four channels: `ndvi` (vegetation vigour, a proxy for how much green biomass there is), `landcover` (the MODIS IGBP class, a proxy for fuel type), `humidity` (relative humidity, a proxy for fine fuel moisture) and `soil_moisture` (a proxy for longer-term drought).

**Weather (Turkish: *hava durumu*).** Wind, air temperature, relative humidity and recent precipitation. Weather is the fastest-changing leg of the triangle — it can change completely in six hours — and it is the reason next-day prediction is a *forecasting* problem rather than a mapping problem.

**In IGNIS**: `air_temp`, `humidity`, `wind_speed`, `wind_u`, `wind_v`, `precip`, and `lst` (land surface temperature).

**Topography (Turkish: *topoğrafya*).** Slope, aspect and elevation. Topography is static, but it interacts strongly with the other two: a south-facing slope in the northern hemisphere receives more solar radiation, so its fuels are drier.

**In IGNIS**: `elevation`, `slope`, `aspect`.

### 1.3 Why fire runs uphill

This is the single most important piece of fire physics for the jury, and it is easy to explain.

Flames lean towards the vertical because hot gas rises (buoyancy, Turkish: *kaldırma kuvveti*). On flat ground the flame stands roughly upright and heats the fuel ahead of it only through radiation across a horizontal gap. On a slope, the ground itself tilts *into* the flame, so the unburnt fuel above the fire sits directly in the rising plume of hot gas. The flame is now effectively lying on top of its next meal.

```
   FLAT GROUND                     UPHILL
                                              🔥
      🔥                                     /  ← fuel above is bathed
     / \    radiation only                  /     in the rising plume
 ___/___\_______fuel___              ______/_____________
                                    /  fuel
```

The consequences are dramatic. As a rule of thumb used in operational fire services, the rate of spread roughly doubles for every 10° of upslope. Rothermel's 1972 spread model — still the basis of most operational fire-behaviour software — encodes this as an explicit slope factor multiplying the reaction intensity term.

**In IGNIS**, this is exactly why `slope` and `aspect` are input channels, and why `elevation` matters. The network is not told the physics; it must *learn* from data that patches with steep slopes and fire on the downhill side tend to have fire uphill tomorrow. Whether it has enough data to learn that is a separate question (see Section 9).

### 1.4 Wind

Wind does three things at once:

1. It tilts the flame forward, putting unburnt fuel into the convective plume — the same mechanism as slope. In fact wind and slope are physically interchangeable to a first approximation; fire-behaviour models often combine them into an "effective wind".
2. It supplies fresh oxygen to the combustion zone, raising the reaction rate.
3. It carries burning embers ahead of the front. This is called **spotting** (Turkish: *sıçrama*) and it is how a fire crosses a road, a river or a firebreak. Spotting is the reason fire spread is not a simple continuous front — new fires appear ahead of the main one.

Wind is a **vector** (Turkish: *vektör*), not a number: it has magnitude *and* direction. This matters enormously for our model design. Knowing that the wind is 12 m/s tells you the fire will move fast; it does not tell you *where*. That is why IGNIS carries three wind channels rather than one:

$$\text{wind\_speed} = \sqrt{u^2 + v^2}$$

where $u$ is the eastward component (positive = wind blowing towards the east) and $v$ is the northward component. The pair $(u, v)$ encodes the direction; `wind_speed` is redundant information, but giving it to the network explicitly saves the network from having to learn the square root.

The vector nature of wind also creates a subtle trap in **data augmentation** (Turkish: *veri artırma*), which we discuss in Section 7.6: if you mirror a patch left-to-right you must also flip the sign of `wind_u`, otherwise you are teaching the network that fires spread against the wind.

### 1.5 Fire regimes

A **fire regime** (Turkish: *yangın rejimi*) is the characteristic pattern of fire in an ecosystem over decades: how often fires occur (frequency), how intense they are, what season they occur in, how large they get, and what type they are.

Fire types worth knowing in English:

| English | Turkish | Meaning |
|---|---|---|
| Ground fire | Toprak altı yangını | Burns in the organic soil layer, slow, smouldering |
| Surface fire | Örtü yangını | Burns litter, grass and shrubs at ground level |
| Crown fire | Tepe yangını | Burns through the tree canopy; fastest and most destructive |
| Spot fire | Sıçrama yangını | New fire ignited by wind-carried embers ahead of the front |

Mediterranean ecosystems have a **crown-fire regime** with a fire return interval on the order of decades. Many Mediterranean plants are not merely fire-tolerant but **fire-adapted**: *Pinus brutia* (Calabrian pine, Turkish: *Kızılçam*) has serotinous cones that require heat to open, so the species actually depends on periodic fire to regenerate. The ecological problem is not fire itself, but fire that is too frequent or too intense for the ecosystem to recover.

### 1.6 Why Türkiye burns

Four factors converge on the Turkish Mediterranean coast — the region around Antalya where you live and where the congress is held.

1. **Climate.** A Mediterranean climate (Köppen *Csa*) means hot dry summers and mild wet winters. Precipitation in July and August is close to zero in Antalya. Fuels therefore dry out for three to four consecutive months every single year. This is why IGNIS restricts its archive to the fire season months of **June to October**.
2. **Fuel.** Türkiye's coastal forests are dominated by *Pinus brutia*, Calabrian pine. It is resinous, it holds large quantities of dead needles in the litter layer, and it carries fire into the crown readily. It is one of the most flammable forest types in the Mediterranean basin.
3. **Topography.** The Taurus Mountains (Turkish: *Toros Dağları*) rise steeply from the coast. Steep slopes accelerate spread as described in Section 1.3, and they also make ground access for firefighting extremely difficult.
4. **Wind.** Summer synoptic patterns bring persistent hot, dry winds. When a heatwave coincides with strong wind, spread rates become uncontrollable by direct attack.

The July–August 2021 fires around **Manavgat and Marmaris** are the reference event: they burned an area on the scale of Türkiye's worst modern fire disaster, forced mass evacuations and required international assistance. As Section 9.6 explains, one of the most serious limitations of our current archive is that **those specific fires are missing from it**.

### 1.7 The four phases of disaster management

Disaster management (Turkish: *afet yönetimi*) is conventionally described as a cycle with four phases. The manuscript uses this framing, so you must be able to state it in English.

| Phase | Turkish | Question it answers | Time scale |
|---|---|---|---|
| **Prevention / Mitigation** | Önleme / Zarar azaltma | How do we reduce the chance and severity of the event? | Years |
| **Preparedness** | Hazırlık | What do we have ready before it happens? | Months |
| **Response** | Müdahale | The event is happening — what do we do *now*? | Hours to days |
| **Recovery** | İyileştirme | How do we rebuild afterwards? | Months to years |

**IGNIS targets the response phase.** This is stated explicitly in Section 1 of the manuscript: *"The work presented here targets the response phase. Once a fire is already burning, the decisive operational question is no longer whether a fire will start but where the fire front will be tomorrow, because that determines the allocation of aircraft, ground crews and evacuation orders."*

Memorise that sentence. It is the one-sentence justification for the whole project, and it separates IGNIS from the very large literature on fire *risk maps*, which serve the prevention and preparedness phases.

### 1.8 Susceptibility is not spread

This distinction is the intellectual core of the project and you must never confuse the two.

| | **Fire susceptibility** | **Fire spread** |
|---|---|---|
| Turkish | Yangın duyarlılığı / riski | Yangın yayılımı |
| Question | Where might a fire start? | A fire is burning — where will it be tomorrow? |
| Temporal? | No — static map | Yes — day $t$ → day $t+1$ |
| Typical predictors | Long-term climate averages, distance to roads, population density, slope, fuel type | Today's fire mask + today's weather + fuel + terrain |
| Target changes | Over years | Every day |
| Literature performance | ROC-AUC often **above 0.93** | Much lower; AUC-PR typically small |
| **Is this our problem?** | **No** | **Yes** |

Why does susceptibility score so well? Because the target is almost static. A steep, dry, remote pine slope near a road is a high-risk pixel this year, next year and the year after. A model can memorise the geography and score highly. There is essentially no forecasting involved.

Spread is different. You must predict which specific pixel, out of roughly four thousand in a patch, will ignite in the next 24 hours. Yesterday's answer does not carry over. As the manuscript puts it, IGNIS "must instead identify which specific pixels, out of roughly four thousand in a patch of which fewer than a dozen are typically burning, will ignite within twenty-four hours."

**If a jury member compares your 0.847 ROC-AUC to a published 0.95 susceptibility score, this table is your answer.** Those are not the same problem and the numbers are not comparable.

#### The formal statement

IGNIS formulates next-day spread as **binary semantic segmentation** (Turkish: *ikili anlamsal bölütleme*).

$$X \in \mathbb{R}^{H \times W \times C}, \qquad H = W = 64, \quad C = 14$$

$$Y \in \{0,1\}^{H \times W}$$

$X$ is the input tensor: a stack of $C$ environmental maps over an $H \times W$ pixel patch, observed on day $t$. $Y$ is the target: a binary mask where 1 means "this pixel was detected as burning on day $t+1$". The model produces

$$P(i,j) = \frac{1}{1 + e^{-z(i,j)}} \in (0,1)$$

a per-pixel probability that pixel $(i,j)$ is burning tomorrow. A binary mask is obtained by thresholding at some $\tau$; the preliminary run used $\tau = 0.5$.

---

## 2. Remote sensing from zero

**Remote sensing** (Turkish: *uzaktan algılama*) means measuring a property of an object without touching it. Your eye is a remote sensor. A satellite is a remote sensor that operates from 700 kilometres away and can see wavelengths your eye cannot.

### 2.1 The electromagnetic spectrum

Light is an electromagnetic wave (Turkish: *elektromanyetik dalga*). Its **wavelength** (Turkish: *dalga boyu*, symbol $\lambda$) determines how it interacts with matter. The full range of wavelengths is the **electromagnetic spectrum** (Turkish: *elektromanyetik tayf*).

```
 short λ ◄───────────────────────────────────────────────────► long λ
 
 Gamma  X-ray   UV │ VISIBLE │  NIR   SWIR  │  MWIR    TIR  │ Microwave  Radio
                   │ 0.4–0.7 │ 0.7–1.3 1.3–3│  3–5     8–14 │
                   │   µm    │     µm    µm │   µm      µm  │
                        ▲        ▲            ▲         ▲
                        │        │            │         │
                     NDVI red  NDVI NIR   fire 4µm   fire 11µm
                                             channel  channel, LST
```

`µm` is a **micrometre**, one millionth of a metre. The units matter for the jury: say "four micrometres", not "four".

The key physical law is **Planck's law** (Turkish: *Planck yasası*), which says every object with a temperature above absolute zero emits radiation, and the *spectrum* of that radiation depends on the temperature. Wien's displacement law gives the peak:

$$\lambda_{\text{peak}} \approx \frac{2898 \ \mu\text{m}\cdot\text{K}}{T}$$

Work through two cases, because this is the entire basis of satellite fire detection:

| Object | Temperature | Peak wavelength |
|---|---|---|
| Normal ground surface | ~300 K (27 °C) | $2898/300 \approx 9.7\ \mu$m — thermal infrared |
| Flaming wildfire | ~800–1000 K | $2898/900 \approx 3.2\ \mu$m — mid-wave infrared |

A fire is not just *brighter* than the background — it is brighter **in a different part of the spectrum**. That difference is what makes automated fire detection possible from orbit.

### 2.2 What a satellite actually records

A satellite image is not a photograph. It is a grid of numbers.

The sensor is an array of detectors. Each detector collects photons arriving from a small solid angle for a short integration time and converts them into an electrical signal, which is digitised into a **digital number** (DN). Ground processing then converts DN into a physical quantity:

- **Radiance** (Turkish: *radyans / ışıma*) — energy per unit area, per unit solid angle, per unit wavelength. Units: W·m⁻²·sr⁻¹·µm⁻¹.
- **Reflectance** (Turkish: *yansıtırlık*) — the fraction of incoming sunlight reflected by the surface, a dimensionless number in [0, 1]. Used for visible/NIR bands.
- **Brightness temperature** (Turkish: *parlaklık sıcaklığı*) — the temperature a perfect black body would need in order to emit the observed radiance. Used for thermal bands, symbol $T_b$.

The correction chain from raw DN to a usable physical value is **radiometric correction** (Turkish: *radyometrik düzeltme*) plus **atmospheric correction** (Turkish: *atmosferik düzeltme*), plus **geometric correction** to place each pixel at the right location on the ground.

**In IGNIS we do none of this ourselves.** All products used are distributed by NASA, ECMWF and UCSB as calibrated, analysis-ready collections. The manuscript states: *"Because the selected products are distributed as calibrated, analysis-ready collections, no additional radiometric correction was applied."* If the jury asks how you did atmospheric correction, the honest answer is: the data providers did it, and we use Level-3 products. That is standard practice and not a weakness.

The final product is a **raster** (Turkish: *raster / hücresel veri*): a two-dimensional array where each cell is a **pixel** (Turkish: *piksel*) covering a fixed ground area, with one value per **band** (Turkish: *bant*).

### 2.3 Passive and active sensing

| | **Passive sensing** | **Active sensing** |
|---|---|---|
| Turkish | Pasif algılama | Aktif algılama |
| Energy source | The Sun, or the Earth's own thermal emission | The sensor emits its own signal |
| Examples | MODIS, Landsat, Sentinel-2, your eye | Radar, LiDAR, SRTM's radar interferometry |
| Works at night? | Only in thermal bands | Yes |
| Sees through cloud? | No (except microwave) | Radar: yes |

IGNIS uses **passive** optical and thermal sensing for everything dynamic (fire, NDVI, LST) and one **active** product for topography (SRTM, which used radar). This mix is worth mentioning because it demonstrates you understand the distinction.

The dependence on passive sensing is also a real limitation: **cloud and thick smoke block the optical and thermal path**. If a fire is under a smoke pall at overpass time, MODIS may not detect it. This is not a hypothetical — Section 9.4 shows it is one of the dominant sources of noise in our target variable.

### 2.4 The three resolutions and the trade-off between them

Every Earth Observation sensor is a compromise between three kinds of resolution.

| Resolution type | Turkish | Definition | MODIS value |
|---|---|---|---|
| **Spatial** | Uzamsal çözünürlük | Ground size of one pixel | 250 m – 1 km |
| **Spectral** | Spektral çözünürlük | Number and narrowness of wavelength bands | 36 bands |
| **Temporal** | Zamansal çözünürlük | How often the same place is revisited | 1–2 times per day |

**Why can't we have all three?** Because of the photon budget. A detector must collect enough photons in its integration time to produce a signal above the electronic noise floor. The number of photons it collects is proportional to (ground area of the pixel) × (width of the spectral band) × (integration time). If you shrink the pixel to get better spatial resolution, you must compensate: widen the spectral band (worse spectral resolution), lengthen the integration time (which means a narrower swath and therefore worse temporal resolution), or accept more noise.

This gives the classic trade-off table:

| Sensor | Spatial | Revisit | Suited to |
|---|---|---|---|
| MODIS (Terra/Aqua) | 250 m – 1 km | 1–2×/day | Daily monitoring of large areas |
| VIIRS (Suomi-NPP) | 375 m (fire) | ~1–2×/day | Finer fire detection |
| Landsat 8/9 | 30 m | 16 days | Detailed mapping, burned-area assessment |
| Sentinel-2 | 10–20 m | 5 days (2 sats) | Detailed vegetation mapping |
| Geostationary (MSG/SEVIRI) | 3 km | Every 15 min | Rapid detection, poor detail |

**IGNIS chose MODIS because next-day prediction requires daily revisit.** Landsat's 30 m detail is useless if the next image is 16 days away — the fire will be over. This is the correct answer to "why not use higher-resolution data?" and you should give it confidently. The honest follow-up, which the manuscript already concedes, is that **VIIRS at 375 m has comparable revisit and would be strictly better**; adopting VIIRS is listed as future work.

### 2.5 Orbits

| Orbit type | Turkish | Altitude | Property |
|---|---|---|---|
| **Low Earth Orbit (LEO)** | Alçak Dünya yörüngesi | 160–2000 km | Fast, close, high detail, narrow view |
| **Sun-synchronous (SSO)** | Güneş eş-zamanlı yörünge | ~700–800 km | Crosses the equator at the *same local solar time* every orbit |
| **Geostationary (GEO)** | Yer durağan yörünge | 35,786 km | Orbital period = 24 h, so it hovers over one longitude |

A **sun-synchronous orbit** is a near-polar orbit whose plane precesses at exactly the rate the Earth orbits the Sun (about 0.986° per day), achieved by choosing an inclination slightly greater than 90° so that the Earth's equatorial bulge produces the required torque. The consequence is that the satellite always observes a given latitude at the same local solar time. This makes images comparable across days — the Sun is always at the same angle — which is essential for detecting *change*, which is exactly what IGNIS does.

A **geostationary orbit** trades spatial detail for temporal density. Meteosat sees all of Europe and Africa every 15 minutes but with 3 km pixels. For rapid fire *detection* that is valuable. For mapping a fire front it is not.

### 2.6 MODIS, Terra and Aqua

**MODIS** = **Mo**derate Resolution **I**maging **S**pectroradiometer. It is an instrument, not a satellite, and there are two copies of it flying on two different NASA satellites:

| Satellite | Launched | Orbit | Local equator crossing |
|---|---|---|---|
| **Terra** | December 1999 | Sun-synchronous, descending | ~10:30 **a.m.** |
| **Aqua** | May 2002 | Sun-synchronous, ascending | ~1:30 **p.m.** |

This is a question you will be asked, so know the answer: **Terra and Aqua are two separate satellites carrying the same instrument, deliberately placed in orbits that observe at different times of day.** Together they provide up to four observations per day of any point on Earth (two daytime, two night-time), because each satellite sees a location on its ascending/descending pass.

MODIS has 36 spectral bands, a 2330 km swath, and spatial resolution of 250 m (bands 1–2), 500 m (bands 3–7) and 1 km (bands 8–36).

**In IGNIS**, we use four MODIS products:

| Product | What it gives | Native resolution |
|---|---|---|
| `MODIS/061/MOD14A1` | Terra daily thermal anomaly / active fire | 1 km / daily |
| `MODIS/061/MYD14A1` | Aqua daily thermal anomaly / active fire | 1 km / daily |
| `MODIS/061/MOD13Q1` | Terra 16-day NDVI composite | 250 m / 16 days |
| `MODIS/061/MOD11A1` | Terra daily land surface temperature | 1 km / daily |
| `MODIS/061/MCD12Q1` | Combined Terra+Aqua annual land cover | 500 m / yearly |

"061" is the **Collection 6.1** processing version. Collections matter: an algorithm change between Collection 5 and Collection 6 changes the numbers. Always cite the collection.

We **merge Terra and Aqua** into a single daily binary fire mask, taking a pixel as burning if either satellite detected fire there that day. This roughly doubles our chance of catching a fire that was obscured or too cool at one overpass.

### 2.7 How a satellite detects a fire

This is the most technically impressive part of Section 2 and it is worth being able to explain in 90 seconds.

**The intuition.** Recall from Section 2.1 that a normal surface at 300 K peaks near 10 µm, while a flaming fire at 800–1000 K peaks near 3–4 µm. Now consider a 1 km MODIS pixel — one million square metres — containing a fire that covers only 1000 m², i.e. 0.1 % of the pixel. In the 11 µm channel the fire barely changes the pixel-average brightness temperature, because the 99.9 % of the pixel that is cool ground dominates. But in the 4 µm channel, the emitted radiance from the hot fraction is enormously larger than the background, because the Planck function rises so steeply with temperature at short wavelengths. So the 4 µm brightness temperature jumps while the 11 µm brightness temperature barely moves.

**The Dozier insight.** Dozier (1981) showed that if you observe a sub-pixel hot spot in two channels at different wavelengths, you have two equations and can solve for two unknowns: the *temperature* of the hot fraction and the *fractional area* it occupies. This is the theoretical foundation of all satellite active-fire detection — you can detect and characterise a fire far smaller than one pixel.

**The MODIS algorithm** (Giglio et al. 2016, Collection 6) works as a **contextual** test. For a candidate pixel it:

1. Computes $T_4$ (brightness temperature at 4 µm) and $T_{11}$ (at 11 µm), and the difference $\Delta T = T_4 - T_{11}$.
2. Applies absolute thresholds to reject obviously cold pixels.
3. Builds a window of **background** pixels around the candidate — valid, cloud-free, non-fire land pixels — and computes their mean and mean absolute deviation of $T_4$ and $\Delta T$.
4. Declares fire if the candidate exceeds the local background by a statistically significant margin, e.g. $T_4 > \bar{T_4} + 3\delta_{T_4}$ and $\Delta T > \overline{\Delta T} + 3\delta_{\Delta T}$.
5. Applies rejection tests for **false alarms**: sun glint off water, desert boundaries, hot bare soil, and coastal pixels.

The word **contextual** matters: the threshold is not fixed, it adapts to the local background. A fire in a cool forest at night is easier to detect than one in a hot desert at noon, and the algorithm accounts for this automatically.

**What this means for us.** MODIS does not report "burned area". It reports **active fire**: a pixel that was actively flaming at the exact moment of the overpass. If the fire was smouldering, or under cloud, or between overpasses, it is not detected. This is the single most important caveat on our target variable and Section 9.4 quantifies its effect.

### 2.8 The FireMask confidence classes

The MOD14A1/MYD14A1 products contain a band called `FireMask` whose values encode the classification of every pixel:

| Value | Meaning |
|---|---|
| 0 | Not processed (missing input data) |
| 1 | Not processed (obsolete) |
| 2 | Not processed (other reason) |
| 3 | Water (non-fire) |
| 4 | Cloud (non-fire) |
| 5 | Non-fire land pixel |
| 6 | Unknown (land pixel, insufficient data) |
| **7** | **Fire — low confidence** |
| **8** | **Fire — nominal confidence** |
| **9** | **Fire — high confidence** |

**IGNIS uses `FireMask >= 7`**, i.e. it accepts low-confidence detections as fire (`FIRE_CONFIDENCE = 7` in `src/config.py`).

This is a design choice with a clear trade-off, and a jury may probe it:

- **Threshold 7** (our choice): maximum **recall** on the fire class. We catch weak and marginal fires. Cost: more false detections in the target mask, which adds label noise.
- **Threshold 9**: maximum **precision**. Only certain fires. Cost: we lose most of the early-stage and small fires — exactly the ones an operational system most needs to predict.

Given that our positive class already accounts for only 0.2686 % of pixels, throwing away two thirds of the positives to raise label purity would make an already impossible learning problem worse. That is the defensible justification.

### 2.9 NDVI

**NDVI** = **N**ormalized **D**ifference **V**egetation **I**ndex (Turkish: *Normalize Edilmiş Fark Bitki Örtüsü İndeksi*).

$$\text{NDVI} = \frac{\rho_{\text{NIR}} - \rho_{\text{Red}}}{\rho_{\text{NIR}} + \rho_{\text{Red}}}$$

where $\rho_{\text{NIR}}$ is reflectance in the near-infrared band (~0.86 µm) and $\rho_{\text{Red}}$ is reflectance in the red band (~0.65 µm).

**Why does this measure vegetation?** Because of two facts about chlorophyll and leaf structure:

1. Chlorophyll **absorbs** red light strongly — that is what it uses for photosynthesis. So healthy leaves have *low* red reflectance, around 0.03–0.05.
2. The internal spongy mesophyll structure of a leaf **scatters** near-infrared light strongly — the leaf is essentially transparent to NIR and bounces it around. So healthy leaves have *high* NIR reflectance, around 0.40–0.50.

The contrast between the two is therefore huge for healthy vegetation, small for stressed vegetation, and reversed for water.

Worked examples:

| Surface | $\rho_{\text{Red}}$ | $\rho_{\text{NIR}}$ | NDVI |
|---|---|---|---|
| Dense healthy forest | 0.04 | 0.45 | $(0.45-0.04)/(0.49) = 0.84$ |
| Dry grass / stressed vegetation | 0.22 | 0.30 | $(0.30-0.22)/(0.52) = 0.15$ |
| Bare soil | 0.25 | 0.28 | $(0.28-0.25)/(0.53) = 0.06$ |
| Water | 0.05 | 0.02 | $(0.02-0.05)/(0.07) = -0.43$ |
| Snow | 0.85 | 0.80 | $(0.80-0.85)/(1.65) = -0.03$ |

NDVI ranges from −1 to +1. Values above about 0.4 indicate substantial green biomass.

**Why the "normalized difference" form?** Dividing by the sum cancels multiplicative effects that scale both bands equally — illumination angle, topographic shading, some atmospheric attenuation. A north-facing slope is darker in both bands, but the *ratio* structure survives. This is why so many remote-sensing indices use the normalized-difference form.

**In IGNIS**, NDVI is our fuel-load proxy. MOD13Q1 is a **16-day composite** at 250 m, so it is not available every day. The pipeline takes the most recent composite within a 32-day window preceding each observation date, and applies the MODIS scale factor of 0.0001 to convert the stored 16-bit integers back into the [−1, 1] range.

**Limitation to state honestly:** NDVI **saturates** at high biomass — once the canopy is closed, adding more leaves does not change NDVI much. It also measures *greenness*, not *dryness*. An index such as NDWI or NDMI, which uses the short-wave infrared band, would be a better fuel-moisture proxy. This is a legitimate improvement to mention as future work.

### 2.10 Land surface temperature

**LST** (Turkish: *arazi yüzey sıcaklığı*) is the radiometric skin temperature of the ground, retrieved from thermal infrared bands (11 and 12 µm on MODIS) using a **split-window algorithm**: the two bands are attenuated differently by atmospheric water vapour, so the difference between them can be used to correct for the atmosphere.

**LST is not air temperature.** This is a favourite jury question and the distinction is real:

| | LST | Air temperature (2 m) |
|---|---|---|
| Turkish | Arazi yüzey sıcaklığı | Hava sıcaklığı |
| What it measures | The temperature of the ground/canopy surface itself | The temperature of the air 2 m above the ground |
| Source in IGNIS | MODIS MOD11A1 (direct observation) | ERA5-Land (model reanalysis) |
| Typical summer noon value | Bare rock can reach 55–60 °C | 35 °C |
| Sensitive to | Surface material, moisture, shading | Air mass, advection |

On a hot dry day, LST over bare soil can exceed air temperature by 20 °C. Over a well-watered irrigated field it can be *below* air temperature, because evaporation cools the surface. That difference is exactly the information we want: it is an indicator of surface moisture stress. **This is why IGNIS carries both `lst` and `air_temp` as separate channels — they are not redundant.**

Because MOD11A1 has cloud gaps, IGNIS computes LST as the **mean of the three preceding daily retrievals**, which substantially reduces missing pixels.

### 2.11 Reanalysis: ERA5-Land

This is one of the questions most likely to be asked at IAC, so understand it properly.

A **reanalysis** (Turkish: *yeniden analiz*) is not an observation and it is not a forecast. It is a *reconstruction of the past state of the atmosphere*, produced by running a modern, frozen-version numerical weather prediction model over historical dates while continuously assimilating every available observation — surface stations, radiosondes, aircraft, ships, buoys, and satellite radiances.

The mechanism is called **data assimilation** (Turkish: *veri özümsemesi*). At each analysis step the model produces a short forecast; the observations available in that window are compared to the forecast; and the model state is nudged towards a statistically optimal compromise between the two, weighted by the estimated error of each. The result is a physically consistent, gap-free, gridded estimate of the atmosphere everywhere, including places and times where nobody measured anything.

**ERA5** is ECMWF's fifth-generation global reanalysis. **ERA5-Land** is a higher-resolution land-surface rerun of ERA5 at approximately 9 km, driven by ERA5 atmospheric forcing (Muñoz-Sabater et al. 2021).

**IGNIS uses `ECMWF/ERA5_LAND/DAILY_AGGR`** — the daily-aggregated version — for five quantities: 2 m air temperature, 2 m dew-point temperature, the eastward wind component $u$, the northward wind component $v$, and volumetric soil water.

> Note for the repository: an earlier version of the code called `ECMWF/ERA5_LAND/DAILY`, which has been removed from Google Earth Engine and produced an `ImageCollection asset ... not found` error. The correct current identifier is `ECMWF/ERA5_LAND/DAILY_AGGR`.

**Is model output a problem?** Give this answer honestly:

*Yes, it is a limitation, and here is the trade-off. Türkiye has a limited number of meteorological stations and none inside a burning forest. Station data is point data — to use it on a 1 km grid you would have to interpolate it, which is itself a model, and a much cruder one than ERA5's data assimilation system. ERA5-Land is physically consistent, gap-free, globally uniform and freely available for the whole historical period we need. Its known weakness is that at ~9 km resolution it cannot represent local terrain-driven winds — valley channelling, slope winds, and the fire's own convectively induced wind field. In steep terrain like the Taurus Mountains, this is a real source of error. A future version could downscale ERA5 with a mesoscale model such as WRF.*

Note the last point especially: **a wildfire generates its own wind**. No reanalysis knows about the fire, so no reanalysis can represent the indraft towards the fire or the plume-driven circulation. This is a fundamental limitation of using any meteorological product for fire spread.

### 2.12 CHIRPS

**CHIRPS** = **C**limate **H**azards Group **I**nfra**R**ed **P**recipitation with **S**tation data (Funk et al. 2015).

CHIRPS is a *blended* precipitation dataset. It combines:
- a high-resolution climatology (the long-term average rainfall pattern);
- satellite thermal-infrared cold-cloud-duration estimates, which infer rainfall from how long cloud tops stay very cold;
- and in-situ rain gauge station data, used to correct the satellite estimates.

It provides daily precipitation on a ~5 km grid from 1981 to near-present.

**Why CHIRPS rather than ERA5 precipitation?** Because precipitation is the field that numerical models get *least* right — it depends on convection at scales the model cannot resolve. A product that is anchored to actual gauge measurements is generally more trustworthy for rainfall than pure model output. Note that IGNIS uses ERA5-Land for temperature/wind/soil moisture but CHIRPS for rain; that is a deliberate choice of the best available source for each variable, and it is worth stating as such.

**Caveat for a fire model:** during the Mediterranean fire season, precipitation over Antalya in July and August is essentially zero. So the `precip` channel is almost always 0 in exactly the situations that matter most, and carries very little information within the fire season. It matters more for representing how long it has been since the last rain — which we do not currently compute. A **days-since-rain** or cumulative-deficit channel would be far more informative than instantaneous daily rainfall, and is a good improvement to propose.

### 2.13 SRTM, elevation, slope and aspect

The **Shuttle Radar Topography Mission** (SRTM, Farr et al. 2007) flew on Space Shuttle Endeavour in February 2000. It carried two radar antennas — one in the payload bay and one on a 60 m deployable mast — and used **interferometric synthetic aperture radar (InSAR)**: by measuring the phase difference between the signals received at the two antennas, the elevation of each point on the ground can be computed by triangulation. In 11 days it mapped roughly 80 % of the Earth's land surface.

The result is a **Digital Elevation Model** (DEM, Turkish: *Sayısal Yükseklik Modeli*): a raster where each pixel value is the height above the reference ellipsoid/geoid in metres. `USGS/SRTMGL1_003` is the 1 arc-second (~30 m) global product.

From a DEM you derive two further variables, both computed by `ee.Terrain.products()` in Google Earth Engine.

**Slope** (Turkish: *eğim*) is the steepness of the surface. Numerically it is the magnitude of the elevation gradient, converted to degrees:

$$\text{slope} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right)$$

In practice the partial derivatives are estimated from a 3×3 neighbourhood of pixels using finite differences — which, as you will see in Section 5.4, is literally a convolution.

**Aspect** (Turkish: *bakı*) is the compass direction the slope faces, measured clockwise from north:

$$\text{aspect} = \arctan2\left(\frac{\partial z}{\partial y}, \frac{\partial z}{\partial x}\right)$$

converted to the range 0–360°. Aspect 0° = the slope faces north; 90° = east; 180° = south; 270° = west.

Aspect matters for fire because in the northern hemisphere a **south-facing slope** receives far more direct solar radiation, so its soils and fuels are hotter and drier. South-facing slopes in the Mediterranean typically carry more flammable, drier fuel.

**A critical numerical warning.** Aspect is a **circular variable** (Turkish: *dairesel değişken*). 359° and 1° describe almost the same direction — they are 2° apart physically — but as raw numbers they differ by 358. If you feed raw aspect into a neural network, the network sees a giant discontinuity at north. **This is one of the confirmed bugs in the current IGNIS model** and Section 7.7 explains how the new pipeline fixes it with a sine/cosine encoding.

A second caveat: SRTM is a 30 m product resampled to our 1 km grid. That resampling averages away most of the fine terrain structure — a 30 m gully that channels fire disappears entirely at 1 km. So our `slope` channel represents broad-scale terrain, not the local slope a firefighter experiences.
---

## 3. Geospatial fundamentals and projections

### 3.1 Why the Earth cannot be flattened

Take an orange, peel it, and try to lay the peel flat on a table without tearing or stretching it. You cannot. This is not a practical difficulty; it is a theorem.

Carl Friedrich Gauss's *Theorema Egregium* (1827) states that the **Gaussian curvature** of a surface is invariant under bending without stretching. A sphere has positive Gaussian curvature everywhere; a plane has zero. Therefore no mapping from sphere to plane can preserve all distances. Every flat map of the Earth distorts something.

A **map projection** (Turkish: *harita projeksiyonu*) is a specific choice about *what* to distort. The possible properties are:

| Property preserved | Name | Turkish | Cost |
|---|---|---|---|
| Angles / local shape | **Conformal** | Konformal / açı koruyan | Areas are distorted |
| Areas | **Equal-area** | Eşit alanlı | Shapes are distorted |
| Distances from a point | **Equidistant** | Eşit uzaklıklı | Only along specific lines |
| Directions from a point | **Azimuthal** | Azimutal | Everything else |

You can preserve *one* of these. Never all.

### 3.2 Geographic versus projected coordinates

**Geographic coordinates** (Turkish: *coğrafi koordinatlar*) are latitude and longitude, measured in **degrees** on the curved surface of an ellipsoid.

- Antalya is at approximately 36.9° N, 30.7° E.
- Problem: **a degree is not a constant distance.** One degree of latitude is always about 111 km. One degree of longitude is 111 km at the equator, about 89 km at Antalya's latitude ($111 \times \cos 36.9° = 88.8$ km), and 0 km at the pole.

This makes lat/lon useless for anything that requires square pixels or area computation. A "0.01° × 0.01°" pixel over Türkiye is 1.11 km tall but only 0.89 km wide — a rectangle, not a square. And a convolutional network sliding a 3×3 kernel over such a grid would be applying a spatially anisotropic operator without knowing it.

**Projected coordinates** (Turkish: *projeksiyonlu koordinatlar*) are x and y measured in **metres** on a flat plane, after applying a projection. Here a 1000 m × 1000 m pixel really is a square kilometre.

**IGNIS uses projected coordinates.** This is the reason the manuscript says the common grid "guarantees that a patch of fixed pixel dimensions always corresponds to a fixed ground area".

### 3.3 UTM and zone 35N

**UTM** = **U**niversal **T**ransverse **M**ercator (Turkish: *Evrensel Enlem Dilimi Merkatör*).

The idea is to accept that a conformal projection distorts badly far from its line of tangency — and then simply never use it far from that line. UTM divides the world into **60 zones**, each 6° of longitude wide, and gives each zone its own transverse Mercator projection centred on that zone's **central meridian**. Within a zone, scale distortion stays below about 1 part in 1000.

Zone numbering starts at 180° W:

$$\text{zone} = \left\lfloor \frac{\text{longitude} + 180}{6} \right\rfloor + 1$$

Apply this and be precise, because it is exactly the sort of detail a jury member may check.

| Zone | Longitude range | Central meridian | Turkish cities |
|---|---|---|---|
| **35N** | 24° E – 30° E | 27° E | İzmir (27.1° E), Muğla / Marmaris (28.3° E), Denizli |
| 36N | 30° E – 36° E | 33° E | Antalya (30.7° E), Manavgat (31.4° E), Ankara, Mersin |
| 37N | 36° E – 42° E | 39° E | Gaziantep, Malatya |
| 38N | 42° E – 48° E | 45° E | Van, Iğdır |

Türkiye spans roughly 26° E to 45° E and therefore genuinely straddles four UTM zones.

So why does IGNIS use **EPSG:32635 = WGS 84 / UTM zone 35N** for the entire country? Because a machine-learning dataset needs **one single grid**. If each location used its "own" zone, patches near a zone boundary would live in different coordinate systems and could not be stacked into a common tensor, and the 1 km pixel grid would not be continuous across the boundary. Choosing one zone for the whole country means accepting larger scale distortion in the east — but our fire archive is dominated by the Aegean and western Mediterranean coasts, which sit in or immediately next to zone 35.

Be ready to present this as an explicit engineering decision with a stated cost, not as an unexamined default. An honest statement of the cost: at 45° E, roughly 18° from zone 35's central meridian, the transverse Mercator scale error becomes large, so any future extension of IGNIS to eastern Anatolia should either switch zones or adopt an equal-area projection such as a Lambert Azimuthal Equal Area centred on Türkiye.

The `N` means **northern hemisphere**; UTM northings are measured from the equator in the north and from a false origin in the south.

### 3.4 EPSG codes

**EPSG** codes are simply unique integer identifiers for coordinate reference systems, maintained in a public registry (originally by the European Petroleum Survey Group). They exist so that "which coordinate system is this?" has a single unambiguous answer.

| EPSG | Name | Units |
|---|---|---|
| 4326 | WGS 84 geographic (lat/lon) | degrees |
| 3857 | WGS 84 / Pseudo-Mercator (web maps) | metres |
| **32635** | **WGS 84 / UTM zone 35N — used by IGNIS** | **metres** |
| 32636 | WGS 84 / UTM zone 36N | metres |

The pattern for UTM north zones is `326` + zone number; south zones are `327` + zone number.

A **coordinate reference system** (CRS) is more than a projection: it is a projection *plus* a **datum** (Turkish: *datum*), which specifies the shape and position of the reference ellipsoid. Using the same coordinates with the wrong datum can shift positions by hundreds of metres. IGNIS uses the WGS 84 datum, the same one GPS uses.

### 3.5 Reprojection and resampling

Our eight products arrive on eight different grids: 250 m MODIS sinusoidal, 500 m sinusoidal, 1 km sinusoidal, ~9 km ERA5 regular lat/lon, ~5 km CHIRPS lat/lon, 30 m SRTM lat/lon. To stack them into a single tensor they must all be placed on the *same* grid.

**Reprojection** (Turkish: *yeniden projeksiyonlama*) is the change of coordinate system. **Resampling** (Turkish: *yeniden örnekleme*) is what happens to the pixel values as a result: the new pixel centres do not fall on old pixel centres, so values must be estimated.

Three standard methods:

| Method | Turkish | How it works | Use when |
|---|---|---|---|
| **Nearest neighbour** | En yakın komşu | Take the value of the closest source pixel | **Categorical data** — land cover, fire masks, any class code |
| **Bilinear** | Çift doğrusal ara değerleme | Weighted average of the 4 surrounding pixels | **Continuous data** — temperature, elevation, NDVI |
| **Cubic convolution** | Kübik evrişim | Weighted combination of the 16 surrounding pixels | Continuous data where smoothness matters; can overshoot |

**The rule you must never break: never use bilinear or cubic on categorical data.** If land cover class 5 is "mixed forest" and class 9 is "savanna", the bilinear average of a 5 and a 9 is 7 — "open shrubland" — which is not between them in any meaningful sense. It is a class that was not there. Similarly, bilinear-interpolating a binary fire mask produces values like 0.37, which is not a valid state of the world.

**In IGNIS**: the fire masks and land cover must use nearest-neighbour; the continuous environmental fields use the default interpolation when reprojecting to 1 km.

There is also a direction question. Going from 30 m SRTM **down** to 1 km is **downsampling / aggregation**: 33×33 ≈ 1089 source pixels contribute to one target pixel, so information is genuinely averaged away. Going from ~9 km ERA5 **up** to 1 km is **upsampling**: we are creating 81 pixels from one, all essentially identical. No new information is created. This is worth stating honestly: **our meteorology is smooth over ~9 km blocks even though we display it at 1 km.** The eight or nine pixels across a patch of 64 km will typically contain only a handful of genuinely distinct ERA5 values.

### 3.6 Co-registration and scale

**Co-registration** (Turkish: *eş-kayıt / çakıştırma*) means guaranteeing that pixel $(i,j)$ in every band refers to the *same piece of ground*. Without it, your NDVI pixel and your fire pixel might be offset by 500 m, and the network would be learning a relationship between a fire and the vegetation of the neighbouring valley.

IGNIS achieves co-registration by reprojecting every band to the same CRS (EPSG:32635) at the same scale (1000 m) with the same grid origin, inside Google Earth Engine. Because GEE handles the grid alignment consistently, all bands stack pixel-for-pixel.

**Scale.** In GEE, "scale" means the nominal ground size of a pixel in metres. `SCALE = 1000` in `src/config.py`. Some consequences worth having at your fingertips:

| Quantity | Value |
|---|---|
| One pixel | 1 km × 1 km = 1 km² = 100 hectares |
| A 65 × 65 patch | 65 km × 65 km = 4,225 km² |
| A 64 × 64 patch (after cropping) | 64 km × 64 km = 4,096 km² |
| A 32 × 32 patch (new pipeline) | 32 km × 32 km = 1,024 km² |
| Mean number of burning pixels per patch today | **12.3**, i.e. ~1,230 hectares |

That last row is worth pausing on. A patch is 4,096 km² and the fire inside it occupies on average 12.3 km². The signal is **0.30 %** of the image. Section 9.5 explains why this is one of the reasons the model fails.

### 3.7 Google Earth Engine

**Google Earth Engine** (GEE, Gorelick et al. 2017) is a cloud platform that hosts a multi-petabyte catalogue of Earth Observation data together with a parallel processing engine, accessible through a JavaScript or Python API. It is the reason six high-school students can build a national-scale multi-sensor dataset without a data centre.

Three GEE concepts you must be able to explain.

**Lazy evaluation (Turkish: *tembel değerlendirme*).** When you write

```python
img = ee.ImageCollection('MODIS/061/MOD14A1').filterDate(d0, d1).max()
```

**no computation happens.** GEE does not download anything. It builds a *description of a computation* — a directed acyclic graph of operations — and holds it. Nothing is computed until you ask for a concrete result. This is why GEE can offer petabytes of data: it only ever computes the pixels you actually request.

**Server-side versus client-side objects.** Anything whose type name begins with `ee.` — `ee.Image`, `ee.Number`, `ee.List`, `ee.Feature` — is a **server-side** object: a handle to a computation that lives on Google's servers. A normal Python `int`, `list` or `float` is **client-side**: it lives in your notebook. The two do not mix freely.

```python
n = ee.Number(5)          # server-side handle
if n > 3:                 # WRONG — Python cannot evaluate a server-side object
    ...
n.getInfo() > 3           # correct, but see below
```

The classic beginner error is writing a Python `for` loop over server-side data. Each iteration triggers a separate round-trip to Google and the notebook takes hours. The correct approach is to use `.map()`, which describes the operation once and lets GEE parallelise it across thousands of machines.

**Why `getInfo()` is slow.** `getInfo()` says: *stop being lazy, execute this entire computation graph now, and send me the answer as a Python object.* That means a synchronous HTTP request, real computation on Google's cluster, and a wait. Every `getInfo()` in a loop multiplies the cost. Furthermore, `getInfo()` has a hard payload limit (on the order of 10 MB / 5000 elements), so it cannot be used to retrieve a real dataset.

**Export tasks.** For anything larger, GEE uses **asynchronous exports**. `Export.table.toDrive(...)` submits a job to a queue; the job runs on Google's infrastructure — possibly for hours — and writes the result into Google Drive or Cloud Storage. Your notebook can be closed in the meantime. This is how IGNIS produces its dataset: 360 daily jobs, each writing one compressed TFRecord shard.

```
   YOUR NOTEBOOK                     GOOGLE'S SERVERS
   ─────────────                     ────────────────
   build ee.Image graph  ──────►     (nothing runs yet)
   
   Export.table.toDrive  ──────►     job queued
        │                                │
        │  notebook may be closed        │ runs for minutes–hours
        ▼                                ▼
   check task status     ◄──────    *.tfrecord.gz written to Drive
```

---

## 4. Machine learning from zero

### 4.1 What machine learning is

Classical programming: a human writes the rules, the computer applies them to data and produces answers.

**Machine learning** (Turkish: *makine öğrenmesi*) inverts this: the human supplies data *and* answers, and the computer finds the rules.

```
  CLASSICAL PROGRAMMING            MACHINE LEARNING
  ─────────────────────            ────────────────
   rules   ──┐                      data    ──┐
             ├──► computer ──► answers        ├──► computer ──► RULES (model)
   data    ──┘                      answers ──┘
```

We use machine learning for fire spread because nobody can write down the rule. A physical model such as Rothermel's exists, but it needs fuel-model parameters, local wind fields and fuel moisture measurements that we do not have at 1 km resolution across a whole country. Machine learning offers a different route: give the algorithm many examples of "these were the conditions, and this is what happened next", and let it find the pattern.

### 4.2 Supervised learning: feature, label, sample

**Supervised learning** (Turkish: *gözetimli öğrenme*) is machine learning where each training example comes with the correct answer attached.

| Term | Turkish | Definition | In IGNIS |
|---|---|---|---|
| **Feature** | Öznitelik | An input variable | One of the 14 (soon 21) channels, e.g. `wind_speed` |
| **Label / target** | Etiket / hedef | The correct answer we want to predict | `fire_next`: was this pixel burning on day $t+1$? |
| **Sample** | Örnek | One (features, label) pair | One 64×64 patch and its next-day mask |
| **Model** | Model | The function mapping features to prediction | The U-Net |
| **Parameters** | Parametreler | The numbers inside the model that get learned | The ~1.9 million weights |
| **Hyperparameters** | Hiperparametreler | Settings chosen by us, not learned | Learning rate 1e-3, batch size 32, $\gamma = 2.0$ |

The full IGNIS training archive is **22,426 samples**, one per fire-centred patch, drawn from **360 fire days** across the 2019, 2020 and 2021 fire seasons.

Note something unusual about our setup: because the label is a *whole mask* rather than a single number, each sample carries 4,096 individual pixel-level labels. So the archive contains roughly **92 million labelled pixels** — a large number that can be misleading, because those pixels are not independent of one another. 4,096 pixels from the same patch share the same weather, the same terrain and the same fire.

### 4.3 Train, validation and test — and why three

You need three separate datasets, and the reason is subtle enough that it is worth spelling out.

| Set | Turkish | Purpose | Touched by |
|---|---|---|---|
| **Training set** | Eğitim kümesi | The model adjusts its parameters to fit this | Gradient descent |
| **Validation set** | Doğrulama kümesi | We choose hyperparameters, epochs, threshold on this | The *human*, repeatedly |
| **Test set** | Test kümesi | Final, one-time honest estimate of performance | Nothing — until the very end |

Why not just two? Because **every time you look at the validation set and change something, you leak a little information from it into your model.** If you try 30 learning rates and pick the one with the best validation score, that best score is optimistically biased — you have effectively fitted the validation set with 30 attempts. The test set exists to give a number that no decision was based on.

**In IGNIS this is a live issue.** The preliminary run split 80 % / 20 % by fire day, and it used the validation set both for early stopping and for reporting. There was no true test set. The new pipeline fixes this with a **year-based split**:

| Split | Years | Purpose |
|---|---|---|
| **Train** | 2019 – 2023 | Fit parameters |
| **Validation** | 2024 | Early stopping, threshold calibration $\tau$, normalisation checks |
| **Test** | 2025 – 2026 | Reported final numbers only |

Why split by *year* rather than randomly? Because of leakage, discussed next.

### 4.4 Data leakage

**Data leakage** (Turkish: *veri sızıntısı*) is when information from the evaluation data influences the model, making the reported score better than the truth.

The dominant form of leakage in remote-sensing datasets is **spatial and temporal autocorrelation**. Consider the naive approach: shuffle all 22,426 patches randomly and take 20 % for validation. On a single fire day, up to 150 patches are drawn from the same fire. They share the same meteorological fields (which are constant over ~9 km ERA5 blocks anyway), the same terrain, and heavily overlapping 64 km footprints. Two such patches are near-duplicates. Put one in training and one in validation and the model can score well by memorising, not by generalising.

**IGNIS avoids this by splitting at the level of whole fire days** — all patches from one day go to the same side of the split. The manuscript is explicit: *"Day-level partitioning eliminates this source of spatial and temporal leakage."* The new year-based split is stricter still: an entire fire season is held out, so even the same fire event cannot appear on both sides.

There is a second, quieter form of leakage that IGNIS's current code commits and the new pipeline fixes: **normalisation statistics computed over the whole dataset.** If you compute the mean and standard deviation of `elevation` using all data including the test years, you have used test data to shape the model's inputs. The correct procedure — implemented in the new pipeline — is to compute $\mu$ and $\sigma$ **from the training split only** and apply those same fixed numbers to validation and test.

### 4.5 Overfitting and underfitting

| | **Underfitting** | **Overfitting** |
|---|---|---|
| Turkish | Yetersiz öğrenme | Aşırı öğrenme / ezberleme |
| Symptom | Bad on training *and* validation | Good on training, bad on validation |
| Cause | Model too simple, trained too little | Model too complex, data too small, trained too long |
| Analogy | A student who did not study | A student who memorised the answers to last year's exam without understanding |

```
  error
    │
    │ ╲                              ╱ validation
    │  ╲                          ╱
    │   ╲___                  ╱
    │       ╲──────────────╱  ← best point: early stopping
    │        ╲___
    │            ╲_________________  training
    └──────────────────────────────────► epochs
       underfit    │ good │   overfit
```

**IGNIS overfits, and we can prove it with numbers.** From the manuscript's Section 4.2:

| Quantity | Training | Validation |
|---|---|---|
| AUC-PR (final epoch) | 0.2375 | 0.0353 |
| AUC-PR (best epoch, #7) | — | **0.0368** |
| Precision | 0.325 | below 0.09 |

Training AUC-PR is **6.7 times** the validation value. That gap is the definition of overfitting. Early stopping halted the run at epoch 25 and restored the epoch-7 weights.

But note the important subtlety, which the manuscript states and which you should be ready to explain: *"the divergence between training and validation curves indicates that the model began memorising individual fire days before it learned generalisable spread behaviour."* Peaking at **epoch 7** is alarmingly early. It suggests the model found something easy to memorise almost immediately. Section 9 argues that the unnormalised inputs are the likely culprit.

### 4.6 The loss function

A **loss function** (Turkish: *kayıp fonksiyonu*), symbol $L$, is a single number measuring how wrong a prediction is. Training means finding parameters that make it small.

For a binary per-pixel prediction, the standard loss is **binary cross-entropy** (BCE, Turkish: *ikili çapraz entropi*):

$$L_{\text{BCE}} = -\big[y \log p + (1-y)\log(1-p)\big]$$

where $y \in \{0,1\}$ is the truth and $p \in (0,1)$ is the predicted probability. Check the behaviour:

| Truth $y$ | Prediction $p$ | Loss | Interpretation |
|---|---|---|---|
| 1 | 0.99 | $-\log 0.99 = 0.010$ | Confident and right → tiny loss |
| 1 | 0.50 | $-\log 0.50 = 0.693$ | Uncertain → moderate loss |
| 1 | 0.01 | $-\log 0.01 = 4.605$ | Confident and wrong → huge loss |
| 0 | 0.01 | $-\log 0.99 = 0.010$ | Confident and right → tiny loss |

The logarithm punishes confident mistakes very heavily, which is the desired behaviour.

**Loss is not the same as the metric.** The loss is what gradient descent optimises; it must be differentiable. The metric (F1, IoU, AUC-PR) is what humans care about; it need not be differentiable. Section 6.10 explains how IGNIS builds losses that are closer to the metric we actually care about.

### 4.7 Gradient descent

Imagine standing on a foggy mountainside, trying to reach the valley floor. You cannot see the valley. But you can feel the slope under your feet. So you take a step in the steepest downhill direction, then feel again, and repeat.

That is **gradient descent** (Turkish: *gradyan inişi*). The loss is the altitude; the parameters are your position; the gradient is the slope.

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

- $\theta$ — the parameters (all ~1.9 million of them)
- $L$ — the loss
- $\nabla_\theta L$ — the **gradient**: the vector of partial derivatives $\partial L / \partial \theta_i$, pointing in the direction of steepest *increase*
- $\eta$ — the **learning rate** (Turkish: *öğrenme oranı*), the step size
- The minus sign makes us go *down*

**The learning rate is the most important hyperparameter you will ever set.**

| Learning rate | Effect | Analogy |
|---|---|---|
| Too small (e.g. $10^{-7}$) | Training is extremely slow; may stall in a shallow dip | Baby steps down a mountain |
| Good (IGNIS: $10^{-3}$) | Steady, reliable descent | Confident walking |
| Too large (e.g. $10$) | Loss oscillates or explodes to NaN | Jumping across the valley and up the other side |

IGNIS uses $\eta = 10^{-3}$, with a **learning-rate schedule**: when validation loss stops improving for 7 epochs, $\eta$ is multiplied by 0.5. This is "take smaller steps as you approach the bottom".

### 4.8 Backpropagation

Gradient descent needs $\partial L / \partial \theta_i$ for every one of the 1.9 million parameters. Computing each one numerically would require 1.9 million forward passes per step — impossible.

**Backpropagation** (Turkish: *geri yayılım*) computes all of them in a *single* backward pass. It is the chain rule of calculus applied systematically.

Conceptually: a neural network is a composition of functions, $L = f_n(f_{n-1}(\dots f_1(X)))$. The chain rule says

$$\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial f_n}\cdot\frac{\partial f_n}{\partial f_{n-1}}\cdots\frac{\partial f_2}{\partial f_1}\cdot\frac{\partial f_1}{\partial \theta_1}$$

Each layer only needs to know (a) the gradient arriving from the layer above and (b) its own local derivative. It multiplies them, keeps what it needs for its own parameters, and passes the rest down.

```
  FORWARD  →  X ──[layer 1]──[layer 2]──[layer 3]──► P ──► L
  BACKWARD ←      ∂L/∂θ₁  ←   ∂L/∂θ₂  ←  ∂L/∂θ₃  ←──────  ∂L/∂P
```

Modern frameworks (TensorFlow, PyTorch) do this automatically — it is called **automatic differentiation** (Turkish: *otomatik türev alma*). You define the forward computation; the framework builds the backward one. You will never write a backpropagation routine by hand, but you must know what it does and be able to say the words "chain rule" if asked.

### 4.9 Epoch, batch, iteration

Three words that are constantly confused. Learn them precisely, because a jury will notice if you misuse them.

| Term | Turkish | Definition |
|---|---|---|
| **Batch** | Yığın / küme | A group of samples processed together in one forward+backward pass |
| **Iteration / step** | Yineleme / adım | One parameter update, using one batch |
| **Epoch** | Devir | One complete pass over the entire training set |

Worked example with IGNIS numbers. Suppose 22,426 patches with an 80/20 split → about 17,940 training patches, batch size 32:

$$\text{iterations per epoch} = \left\lceil \frac{17{,}940}{32} \right\rceil = 561$$

So one epoch = 561 parameter updates. Training was configured for a maximum of 120 epochs, i.e. up to 67,320 updates, but early stopping ended it at epoch 25.

**Why batches at all?** Two reasons:
1. **Memory.** All 17,940 patches at once would not fit in GPU memory (see Section 8.4).
2. **Noise helps.** Updating on a small random subset makes each gradient a noisy estimate of the true gradient. That noise helps the optimiser escape shallow local minima. This is why the method is called **stochastic** gradient descent.

### 4.10 Optimisers

| Optimiser | Turkish | Idea | Note |
|---|---|---|---|
| **SGD** | Stokastik gradyan inişi | The plain update rule of Section 4.7 | Simple, needs careful tuning |
| **SGD + momentum** | Momentumlu SGD | Accumulate a velocity so you keep rolling through flat regions | Like a ball with mass |
| **Adam** | — | Per-parameter adaptive learning rates from running estimates of the first and second moments of the gradient | **IGNIS uses this** |
| **AdamW** | — | Adam with *decoupled* weight decay, which is the mathematically correct way to apply L2 regularisation with adaptive methods | Recommended for the new pipeline |

**Adam** (Kingma & Ba, 2015 — the reference cited in the manuscript) maintains for each parameter:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment: mean of gradients)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second moment: mean of squared gradients)}$$

and updates

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

with bias-corrected $\hat m, \hat v$. The intuition: a parameter whose gradient has been consistently large gets a *smaller* effective step (because $\sqrt{v}$ is large), and a parameter whose gradient is small and consistent gets a *larger* one. Every parameter effectively gets its own learning rate. Typical defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### 4.11 Regularisation

**Regularisation** (Turkish: *düzenlileştirme*) is anything you do to reduce overfitting.

**Dropout (Turkish: *seyreltme*).** During training, randomly set a fraction $p$ of activations to zero at each step. IGNIS uses $p = 0.2$. Analogy: a football team that trains with two random players missing each session — every player must learn to cover, so the team stops depending on any single star. At inference time dropout is turned off and activations are scaled to compensate.

**Batch normalisation (Turkish: *yığın normalizasyonu*).** For each channel, normalise the activations across the batch to zero mean and unit variance, then apply a learned scale $\gamma$ and shift $\beta$:

$$\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}, \qquad y = \gamma \hat{x} + \beta$$

This keeps activations in a well-behaved range throughout the network, allows larger learning rates, and adds a small regularising noise. **IGNIS applies batch normalisation after every convolution.**

An important observation for Section 9: batch normalisation normalises *inside* the network. It does **not** normalise the *inputs*. Our raw `elevation` channel with a standard deviation of 515.44 hits the very first convolution before any BatchNorm layer exists. That first layer is precisely where the damage is done.

**Weight decay (Turkish: *ağırlık azaltma*).** Add a penalty $\lambda \sum \theta_i^2$ to the loss, which pushes weights towards zero unless the data strongly justifies them. Equivalent to L2 regularisation.

**Early stopping (Turkish: *erken durdurma*).** Monitor a validation metric; when it stops improving for `patience` epochs, stop and restore the best weights. IGNIS monitors validation AUC-PR with patience 18. In the preliminary run it stopped at epoch 25, restoring epoch 7.

**Data augmentation (Turkish: *veri artırma*).** Create new training samples by applying label-preserving transformations. Section 7.8 covers the direction-aware augmentation IGNIS needs.

---

## 5. Deep learning and convolutional networks

### 5.1 The artificial neuron

The basic unit computes a weighted sum of its inputs, adds a bias, and passes the result through a nonlinear function.

$$y = \sigma\!\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

- $x_i$ — inputs
- $w_i$ — weights (learned)
- $b$ — bias (learned)
- $\sigma$ — activation function

```
   x₁ ──w₁──┐
   x₂ ──w₂──┤
   x₃ ──w₃──┼──►  Σ  ──► +b ──► σ(·) ──► y
   ...      │
   xₙ ──wₙ──┘
```

A **layer** is many neurons in parallel; a **deep network** is many layers stacked. "Deep learning" (Turkish: *derin öğrenme*) simply means neural networks with many layers.

### 5.2 Activation functions

**Why is a nonlinearity necessary?** Because without it the whole network collapses. If layer 1 computes $W_1 x$ and layer 2 computes $W_2(W_1 x)$, the result is $(W_2 W_1)x = W'x$ — a single linear transformation. A hundred stacked linear layers are exactly as expressive as one. All the depth is wasted.

| Function | Formula | Range | Where used in IGNIS |
|---|---|---|---|
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | After every convolution in the encoder, bottleneck and decoder |
| **Sigmoid** | $1/(1+e^{-x})$ | $(0, 1)$ | The final output layer, converting a score to a probability |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $(-1, 1)$ | Not used here |
| Leaky ReLU | $\max(0.01x, x)$ | $\mathbb{R}$ | Alternative, avoids "dead" neurons |

**ReLU** (Turkish: *düzeltilmiş doğrusal birim*) is popular because it is trivially cheap and because its derivative is exactly 1 for positive inputs, which avoids the "vanishing gradient" problem that plagued deep sigmoid networks. Its weakness is the "dying ReLU": a neuron whose input is always negative outputs 0 forever and receives zero gradient, so it never recovers.

**Sigmoid** maps any real number to $(0,1)$, which is what we need at the output — a probability.

| $z$ | $\sigma(z)$ |
|---|---|
| −5 | 0.0067 |
| −2 | 0.1192 |
| 0 | 0.5000 |
| 2 | 0.8808 |
| 5 | 0.9933 |

Note that $\sigma(0) = 0.5$ exactly. This is why the default threshold $\tau = 0.5$ corresponds to "the raw score $z$ is positive". Section 6.8 explains why 0.5 is a terrible threshold for an imbalanced problem.

### 5.3 Why fully connected layers fail on images

Suppose we ignored convolutions and used a **fully connected** (dense) layer directly on our input patch. The input is $64 \times 64 \times 14 = 57{,}344$ numbers. Connect it to a modest hidden layer of 1,000 neurons:

$$57{,}344 \times 1{,}000 + 1{,}000 = 57{,}345{,}000 \text{ parameters}$$

**57 million parameters in a single layer** — thirty times the entire U-Net — and we have not even started building the network. This is the **parameter explosion** problem.

The second problem is worse. A fully connected layer has a *separate weight for every input position*. It learns "the value at pixel (17, 42) matters" — not "a fire next to dry fuel matters". If you shift the fire two pixels to the right, every weight is now wrong. The layer has no notion that translated patterns are the same pattern. This is **position sensitivity** — the absence of *translation equivariance*.

Convolution solves both problems at once:

| Problem | Convolution's solution |
|---|---|
| Parameter explosion | **Weight sharing**: the same small kernel is used at every position |
| Position sensitivity | **Translation equivariance**: shift the input, the output shifts identically |
| Ignoring spatial structure | **Locality**: each output depends only on a small neighbourhood |

Compare: a 3×3 convolution from 14 channels to 32 channels has $9 \times 14 \times 32 + 32 = 4{,}064$ parameters — about **14,000 times fewer** than that dense layer — and it works at every position in the image.

### 5.4 The convolution, worked by hand

A **convolution** (Turkish: *evrişim*) slides a small matrix — the **kernel** or **filter** (Turkish: *çekirdek / filtre*) — over the input, and at each position computes the sum of element-wise products.

$$(I * K)(i,j) = \sum_{m}\sum_{n} I(i+m,\ j+n)\, K(m,n)$$

Let us do this with real numbers.

#### Example input: a small fire mask

Take a 5×5 patch of the `fire` channel. 1 = burning, 0 = not burning.

```
        col:  0   1   2   3   4
    row 0 [   0   0   0   0   0 ]
    row 1 [   0   0   1   0   0 ]
    row 2 [   0   1   1   1   0 ]
    row 3 [   0   0   1   0   0 ]
    row 4 [   0   0   0   0   0 ]
```

A small cross-shaped fire of 5 burning pixels.

#### Kernel A: "count my burning neighbours"

```
    K_A =  [ 1  1  1 ]
           [ 1  0  1 ]
           [ 1  1  1 ]
```

**Output at position (2,2).** The 3×3 window centred on (2,2) covers rows 1–3, columns 1–3:

```
    window = [ 0  1  0 ]
             [ 1  1  1 ]
             [ 0  1  0 ]
```

Multiply element by element with $K_A$ and sum:

$$(0{\cdot}1) + (1{\cdot}1) + (0{\cdot}1) + (1{\cdot}1) + (1{\cdot}0) + (1{\cdot}1) + (0{\cdot}1) + (1{\cdot}1) + (0{\cdot}1)$$
$$= 0 + 1 + 0 + 1 + 0 + 1 + 0 + 1 + 0 = \mathbf{4}$$

The centre pixel has 4 burning neighbours.

**Output at position (1,2).** Window covers rows 0–2, columns 1–3:

```
    window = [ 0  0  0 ]
             [ 0  1  0 ]
             [ 1  1  1 ]
```

$$0+0+0+0+(1{\cdot}0)+0+1+1+1 = \mathbf{3}$$

Doing this at every valid position gives the full output:

```
    positions (1,1)…(3,3):
         [ 2   2   2 ]
         [ 3   4   3 ]
         [ 2   2   2 ]
```

**Interpretation:** this single 3×3 kernel has computed, for every pixel, how much fire surrounds it. That is a fire-density feature — and it is exactly the kind of quantity that determines whether a pixel will ignite tomorrow. **One 9-number kernel, applied everywhere, extracts a physically meaningful feature.** That is the whole idea of a CNN.

#### Kernel B: "which side is the fire on?" — a directional kernel

```
    K_B =  [ -1   0   +1 ]
           [ -2   0   +2 ]
           [ -1   0   +1 ]
```

This is the horizontal Sobel operator, an edge detector.

**Output at position (2,1)** — window covers rows 1–3, columns 0–2:

```
    window = [ 0  0  1 ]
             [ 0  1  1 ]
             [ 0  0  1 ]
```

$$(0{\cdot}{-1}) + (0{\cdot}0) + (1{\cdot}{+1}) + (0{\cdot}{-2}) + (1{\cdot}0) + (1{\cdot}{+2}) + (0{\cdot}{-1}) + (0{\cdot}0) + (1{\cdot}{+1})$$
$$= 1 + 2 + 1 = \mathbf{+4}$$

**Output at position (2,3)** — window covers rows 1–3, columns 2–4:

```
    window = [ 1  0  0 ]
             [ 1  1  0 ]
             [ 1  0  0 ]
```

$$(1{\cdot}{-1}) + 0 + 0 + (1{\cdot}{-2}) + 0 + 0 + (1{\cdot}{-1}) + 0 + 0 = -1 -2 -1 = \mathbf{-4}$$

**Interpretation:** the *sign* of the output tells you which side the fire is on. Positive means "fire is to my east"; negative means "fire is to my west". A directional kernel like this, combined with the `wind_u` channel, is precisely the mechanism by which a CNN could learn "fire spreads downwind". Whether ours *does* learn it is an open question — a **channel-ablation study** (Section 10, Q22) would tell us.

#### The crucial point

Nobody wrote $K_A$ or $K_B$ into IGNIS. **The kernel values are parameters.** They start as random numbers and gradient descent adjusts them until they extract whatever features minimise the loss. The kernels above are illustrations of the *kind* of thing a trained network discovers. In practice, the first layer of a trained CNN typically learns edge and gradient detectors that look remarkably like Sobel operators — the network rediscovers them because they are useful.

### 5.5 Stride, padding and the receptive field

**Stride (Turkish: *adım*)** is how far the kernel jumps between positions. Stride 1 (IGNIS's choice for all convolutions) evaluates every position. Stride 2 skips every other position and halves the output size.

**Padding (Turkish: *dolgu*).** In the example above, a 5×5 input with a 3×3 kernel gave only a 3×3 output — the border pixels have no complete neighbourhood. In general:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

| Padding mode | Turkish | Effect |
|---|---|---|
| `valid` (p = 0) | Dolgusuz | Output shrinks by $k-1$ each layer |
| `same` (p = (k−1)/2) | Aynı boyut | Output same size as input; border filled with zeros |

**IGNIS uses `same` padding.** With a 3×3 kernel, $p=1$: one ring of zeros around the input. This keeps every feature map at 64×64 within a level, which is what allows the decoder's skip connections to concatenate cleanly. The cost is a mild border artefact — the outermost pixels see artificial zeros.

**Receptive field (Turkish: *alıcı alan*)** is the region of the *original input* that influences one output value. This is the concept that tells you how much geographic context your network can actually use.

A single 3×3 convolution: receptive field 3×3. Stack two: 5×5. Add a 2×2 pooling and the receptive field doubles in effect. Let us compute it for the actual IGNIS encoder:

| Stage | Operation | Receptive field | Ground extent |
|---|---|---|---|
| Input | — | 1×1 | 1 km |
| Level 1 | two 3×3 convs | 5×5 | 5 km |
| ↓ pool /2 | | | |
| Level 2 | two 3×3 convs | 13×13 | 13 km |
| ↓ pool /4 | | | |
| Level 3 | two 3×3 convs | 29×29 | 29 km |
| ↓ pool /8 | | | |
| Bottleneck | two 3×3 convs | **61×61** | **61 km** |

So a single neuron in the IGNIS bottleneck integrates information from a **61 km × 61 km** region — essentially the whole 64 km patch. Architecturally, therefore, the network *can* see the global weather context of the patch. This is a good fact to have ready: it shows the depth of 3 was not chosen arbitrarily, it is exactly enough to cover the patch.

### 5.6 Pooling

**Pooling** (Turkish: *havuzlama*) reduces spatial size by summarising each small block.

**Max pooling 2×2** takes the maximum of each non-overlapping 2×2 block:

```
   input 4×4            output 2×2
   [ 1  3 | 2  4 ]      [ 3  4 ]
   [ 2  0 | 1  0 ]  →   [ 6  9 ]
   ─────── ───────
   [ 5  6 | 8  9 ]
   [ 1  2 | 7  3 ]
```

Three effects: (1) it halves each spatial dimension, cutting computation by 4×; (2) it doubles the effective receptive field of everything downstream; (3) it introduces a small amount of translation *invariance* — a feature that moves by one pixel often produces the same pooled output.

Effect (3) is a double-edged sword for segmentation. We *want* precise localisation of the fire boundary, and pooling deliberately discards exactly that. This tension is what skip connections exist to resolve (Section 5.9).

**In IGNIS:** 2×2 max pooling after each encoder block, so the spatial size goes $64 \to 32 \to 16 \to 8$ and the bottleneck operates on an 8×8 grid.

### 5.7 Channels

A **channel** (Turkish: *kanal*) is one 2D map in a stack. A normal photograph has 3 channels: red, green, blue. Our input patch has **14 channels**, each a different physical quantity:

```
                    64
              ┌──────────────┐
           ┌──┴───────────┐  │  ndvi
        ┌──┴───────────┐  │  │  lst
     ┌──┴───────────┐  │  │  │  air_temp
   64│              │  │  │  │  ... 14 layers total ...
     │  each layer  │  │  │  │  landcover
     │  is 64 × 64  │  │  ├──┘  fire
     └──────────────┘  ├──┘
                       └─────  → tensor shape (64, 64, 14)
```

A convolution operating on a multi-channel input has a kernel with a channel dimension too: a 3×3 kernel applied to a 14-channel input is really a $3 \times 3 \times 14$ block of 126 weights. It sums across *all* channels simultaneously. This is the key to how a CNN combines physics: a single kernel can implement "high wind AND low humidity AND fire nearby" by giving positive weight to `wind_speed`, negative weight to `humidity`, and positive weight to neighbouring `fire` pixels.

To get 32 output channels you use 32 such kernels, each learning a different combination. That is $32 \times (3 \times 3 \times 14) + 32 = 4{,}064$ parameters, exactly as computed in Section 5.3.

The tensor shape convention in TensorFlow/Keras is `(batch, height, width, channels)` — "channels last". PyTorch uses `(batch, channels, height, width)` — "channels first". **The new IGNIS pipeline is PyTorch, so remember: (N, C, H, W).** For a batch of 32 patches with 21 channels at 32×32, that is `(32, 21, 32, 32)`.

### 5.8 Classification versus semantic segmentation

| | **Classification** | **Semantic segmentation** |
|---|---|---|
| Turkish | Sınıflandırma | Anlamsal bölütleme |
| Input | An image | An image |
| Output | One label for the whole image | One label for **every pixel** |
| Example | "This patch contains a growing fire" | "These 47 pixels will burn tomorrow" |
| Output shape | a vector of $K$ numbers | $H \times W$ map |

IGNIS does **both**, in sequence, and the manuscript is explicit about why this two-level design was chosen:

1. The U-Net performs **semantic segmentation**, producing a 64×64 probability map.
2. A post-processing rule reduces that map to a **patch-level class** — growing, stable or extinguishing — by counting predicted fire pixels:

$$r = \frac{N_{t+1}}{\max(N_t,\ 1)}$$

$$r > 1.25 \Rightarrow \text{growing}; \qquad 0.75 \le r \le 1.25 \Rightarrow \text{stable}; \qquad r < 0.75 \Rightarrow \text{extinguishing}$$

The $\max(N_t, 1)$ in the denominator prevents division by zero.

Why the second step? Because, as the manuscript puts it, *"Operational decision-makers require a categorical statement rather than a probability field."* A duty officer allocating aircraft wants "this fire is growing", not a 4,096-element array of probabilities.

Why did the segmentation come first, rather than training a classifier directly? Two reasons given in the manuscript: *"The direction of expansion and the extent of the burned area are predicted jointly, and the spatial continuity of the fire front is preserved, because the loss is evaluated on every pixel rather than on an aggregate statistic."*

**Note the threshold change.** An earlier version used $r > 1.15$ for growing and $0.85$ as the stable lower bound (as still documented in the repository README). The manuscript widened the stable band to $[0.75, 1.25]$ because *"the stable class so rare that it was effectively unlearnable"*. If a jury member compares the README and the paper, this is the explanation.

### 5.9 The U-Net

The **U-Net** was introduced by Ronneberger, Fischer and Brox in 2015 for biomedical image segmentation. It is now the default architecture for dense prediction on small datasets, and it is what IGNIS uses.

```
 INPUT 64×64×14
      │
 ┌────▼─────────────┐                                 ┌───────────────────┐
 │ E1: conv3×3 ×2   │─────── skip connection ────────►│ D1: concat + conv │──► 1×1 conv
 │     32 filters   │           (64×64×32)            │     32 filters    │      sigmoid
 │  64×64×32        │                                 │   64×64×32        │        │
 └────┬─────────────┘                                 └───────▲───────────┘        ▼
      │ maxpool 2×2                                           │ upconv 2×2   64×64×1
 ┌────▼─────────────┐                                 ┌───────┴───────────┐   PROBABILITY
 │ E2: conv3×3 ×2   │─────── skip connection ────────►│ D2: concat + conv │      MAP
 │     64 filters   │           (32×32×64)            │     64 filters    │
 │  32×32×64        │                                 │   32×32×64        │
 └────┬─────────────┘                                 └───────▲───────────┘
      │ maxpool 2×2                                           │ upconv 2×2
 ┌────▼─────────────┐                                 ┌───────┴───────────┐
 │ E3: conv3×3 ×2   │─────── skip connection ────────►│ D3: concat + conv │
 │    128 filters   │          (16×16×128)            │    128 filters    │
 │  16×16×128       │                                 │   16×16×128       │
 └────┬─────────────┘                                 └───────▲───────────┘
      │ maxpool 2×2                                           │ upconv 2×2
      │              ┌─────────────────────────┐              │
      └─────────────►│ BOTTLENECK              │──────────────┘
                     │ conv3×3 ×2, 256 filters │
                     │        8×8×256          │
                     └─────────────────────────┘

           ENCODER (contracting)          DECODER (expanding)
           "WHAT is in the patch?"        "WHERE exactly?"
```

The shape of that diagram — down the left, across the bottom, up the right — is why it is called a **U**-Net.

#### Why the encoder compresses

Each encoder level halves the spatial resolution and doubles the number of channels. The network trades *spatial* information for *semantic* information. At 64×64×14 the network holds raw physical measurements at fine spatial detail. At 8×8×256 it holds 256 highly abstract feature maps at coarse spatial detail — things like "steep dry south-facing terrain with strong wind from the west and an active front".

This is necessary because a wide receptive field is only obtainable by downsampling (or by many more layers). To know whether this pixel will burn, the network must know what is happening tens of kilometres away.

#### The bottleneck

The **bottleneck** (Turkish: *darboğaz*) is the deepest, most abstract representation: 8×8 spatial positions × 256 channels = 16,384 numbers, compared to 57,344 in the input. It contains the network's global understanding of the scene, and it holds more than half the model's parameters (885,248 of ~1.93 million — see Section 5.10).

#### Why the decoder uses transposed convolutions

The decoder must get back from 8×8 to 64×64. Simple upsampling (repeating each pixel) works, but it is a fixed operation with no parameters. A **transposed convolution** (Turkish: *ters evrişim*, also called deconvolution or up-convolution) is a *learnable* upsampling: it inserts zeros between input pixels and then convolves, so the network learns how to expand each feature.

```
   input 2×2         insert zeros (stride 2)      convolve with learned 2×2
   [ a  b ]          [ a  0  b  0 ]               → 4×4 learned output
   [ c  d ]     →    [ 0  0  0  0 ]        →
                     [ c  0  d  0 ]
                     [ 0  0  0  0 ]
```

In IGNIS each decoder level uses a 2×2 transposed convolution with stride 2, which doubles the spatial size and halves the channel count.

A known artefact: transposed convolutions can produce a **checkerboard pattern** when the kernel size is not divisible by the stride. With 2×2 kernel and stride 2 this is avoided, so IGNIS is safe here.

#### What skip connections do — and why they matter for a fire front

This is the most important part of Section 5. If you understand one thing about the U-Net, understand this.

**The problem.** The encoder throws away spatial precision. After three 2×2 max-pools, each bottleneck pixel represents 8×8 = 64 input pixels, i.e. an 8 km × 8 km block. When the decoder upsamples back to 64×64, it can only produce *blurry* structures — it cannot know exactly which of those 64 pixels the boundary was in. The information is simply gone.

**The solution.** A **skip connection** (Turkish: *atlama bağlantısı*) copies the encoder's feature map *before* pooling and concatenates it to the corresponding decoder feature map. So the decoder receives two things:

- from below: **semantic** information ("there is a growing fire front here, driven by west wind on a steep slope") — accurate about *what*, vague about *where*;
- from the side: **spatial** information at full resolution ("here is exactly where the boundary pixel is") — accurate about *where*, ignorant about *what*.

The subsequent convolution learns to combine them.

```
   Without skip connections:          With skip connections:

     true fire front                    true fire front
     ████                               ████
     ████                               ████

     prediction (blurry)                prediction (sharp)
     ▒▒▒▒▒▒                             ████
     ▒▒▒▒▒▒                             ████
     ▒▒▒▒▒▒
```

**Why this is vital for IGNIS specifically.** A fire front is a *thin* structure. Our fires occupy on average 12.3 pixels out of 4,096 — 0.30 % of the patch. The whole quantity of interest is a handful of pixels wide. An architecture that blurs boundaries by 8 pixels would destroy the entire signal. The manuscript states this exactly: *"The skip connections reinstate the fine spatial detail that successive downsampling would otherwise discard, which is essential for delineating a fire boundary that may be only a few pixels wide."*

A second, more technical benefit: skip connections provide a short path for gradients to flow from the loss back to the early encoder layers, which makes deep networks easier to train — the same principle as residual connections in ResNet.

#### The formal equations

The manuscript states the architecture as:

$$f_l = \text{Pool}\big(\sigma(\text{Conv}(f_{l-1}))\big), \qquad l = 1 \dots L$$

$$g_{l-1} = \sigma\big(\text{Conv}([\,\text{Up}(g_l),\ f_{l-1}\,])\big)$$

where $f_l$ is the encoder feature map at level $l$, $g_l$ the decoder feature map, $[\cdot,\cdot]$ channel-wise concatenation, $\text{Up}(\cdot)$ the transposed convolution, $\sigma$ the ReLU, and $f_0 = X$. The output is

$$P(i,j) = \frac{1}{1+\exp(-z(i,j))} \in (0,1)$$

### 5.10 Counting the parameters

You should be able to count parameters, because a jury may ask "how big is your model?" and "why so small?".

**Rules.**

| Layer | Parameter count |
|---|---|
| $k \times k$ conv, $C_{\text{in}} \to C_{\text{out}}$ | $k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}$ |
| $k \times k$ transposed conv, $C_{\text{in}} \to C_{\text{out}}$ | $k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}$ |
| BatchNorm over $C$ channels | $2C$ trainable ($\gamma, \beta$) + $2C$ non-trainable (running $\mu, \sigma$) |
| Max pooling, ReLU, dropout | **0** |

**Worked example — the very first convolution.** 3×3, from 14 input channels to 32 output channels:

$$9 \times 14 \times 32 + 32 = 4{,}032 + 32 = 4{,}064$$

**The whole IGNIS U-Net:**

| Block | Operation | Parameters |
|---|---|---|
| E1 | conv 14→32; conv 32→32 | 4,064 + 9,248 = **13,312** |
| E2 | conv 32→64; conv 64→64 | 18,496 + 36,928 = **55,424** |
| E3 | conv 64→128; conv 128→128 | 73,856 + 147,584 = **221,440** |
| Bottleneck | conv 128→256; conv 256→256 | 295,168 + 590,080 = **885,248** |
| D3 | upconv 256→128; conv 256→128; conv 128→128 | 131,200 + 295,040 + 147,584 = **573,824** |
| D2 | upconv 128→64; conv 128→64; conv 64→64 | 32,832 + 73,792 + 36,928 = **143,552** |
| D1 | upconv 64→32; conv 64→32; conv 32→32 | 8,224 + 18,464 + 9,248 = **35,936** |
| Output | 1×1 conv 32→1 | 32 + 1 = **33** |
| BatchNorm | 14 layers, 1,408 channels total | **2,816** trainable |
| | **TOTAL** | **≈ 1,931,585** |

This reproduces the **≈1.9 million** figure reported in the manuscript, and the table shows something worth saying out loud: **46 % of the parameters are in the bottleneck alone.** The two 256-filter convolutions cost 885,248 parameters. That is where the model's capacity lives.

Note also the concatenation in the decoder: `upconv 256→128` produces 128 channels, which are concatenated with the 128 skip channels from E3, giving 256 input channels to the next conv — hence `conv 256→128`.

**With the new 21-channel input**, only the first convolution changes:

$$9 \times 21 \times 32 + 32 = 6{,}048 + 32 = 6{,}080$$

an increase of 2,016 parameters — about 0.1 % of the model. The lesson is important and worth stating to a jury: **adding well-engineered input channels is almost free in parameter terms, while adding filters in the bottleneck is very expensive.** Effort spent on input representation has a far better cost/benefit ratio than effort spent making the network bigger.
---

## 6. Class imbalance and metrics

**Read this section twice.** It is the scientific heart of IGNIS. Almost every hard question you will be asked at IAC comes from here, and almost every mistake made in the wildfire machine-learning literature is a mistake in this section.

The situation is easy to state. In our archive, **0.2686 %** of pixels are positive (burning tomorrow). That is roughly **11 pixels out of 4,096** in a typical patch. Everything that follows is a consequence of that single number.

### 6.1 The confusion matrix

For a binary problem, every prediction falls into one of four boxes.

```
                          PREDICTED
                    Fire            No fire
                ┌───────────────┬───────────────┐
         Fire   │      TP       │      FN       │
                │ True Positive │ False Negative│
                │  (doğru poz.) │ (yanlış neg.) │
OBSERVED        ├───────────────┼───────────────┤
       No fire  │      FP       │      TN       │
                │False Positive │ True Negative │
                │ (yanlış poz.) │ (doğru neg.)  │
                └───────────────┴───────────────┘
```

| Box | Turkish | Meaning | Operational consequence in IGNIS |
|---|---|---|---|
| **TP** | Doğru pozitif | We said fire, it burned | Correct warning — resources sent to the right place |
| **FP** | Yanlış pozitif | We said fire, it did not burn | **False alarm** — aircraft and crews wasted |
| **FN** | Yanlış negatif | We said no fire, it burned | **Missed fire** — an unprotected village |
| **TN** | Doğru negatif | We said no fire, it did not burn | Correct, and utterly uninformative here |

For a wildfire system, FN and FP have wildly different costs. A missed fire can kill people. A false alarm wastes fuel and flight hours. The asymmetry is real and it should influence which metric you optimise (see Section 6.10, Tversky loss).

### 6.2 Why accuracy lies

**Accuracy** (Turkish: *doğruluk*) is the fraction of predictions that are correct:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Now apply it to our data. The archive contains 22,426 patches × 4,096 pixels ≈ **91,856,896 pixels**. At a prevalence of 0.2686 %:

| Quantity | Count |
|---|---|
| Total pixels | 91,856,896 |
| Positive pixels (burning tomorrow) | ≈ 246,730 |
| Negative pixels | ≈ 91,610,166 |

Consider **the null model**: a program consisting of one line, `return 0`. It predicts "no fire" for every pixel in Türkiye, forever. Its confusion matrix is:

| | Predicted fire | Predicted no fire |
|---|---|---|
| **Observed fire** | TP = 0 | FN = 246,730 |
| **Observed no fire** | FP = 0 | TN = 91,610,166 |

$$\text{Accuracy} = \frac{0 + 91{,}610{,}166}{91{,}856{,}896} = 0.997314 = \mathbf{99.73\ \%}$$

**A program that has never heard of fire scores 99.73 % accuracy.**

Its precision is undefined (0/0), its recall is 0, its F1 is 0, its IoU is 0, and it is completely useless. But the accuracy number looks magnificent. If you saw "99.7 % accuracy" in a paper abstract without the prevalence stated, you would have no way to know whether the model is excellent or is literally the constant function zero.

**This is why the manuscript says:** *"Because the positive class is extremely rare, overall pixel accuracy is uninformative: predicting no fire anywhere already yields more than 99.7 % accuracy."*

The general rule to memorise and to say out loud at IAC:

> **A metric is only interpretable relative to its baseline. Never report a metric without also reporting what a trivial predictor would score.**

We will see in Section 6.7 and Section 9 that IGNIS is itself an example of a project that violated this rule at the patch level and got caught — by its own authors.

### 6.3 Precision, recall, F1

These three metrics ignore TN entirely, which is exactly why they are useful here.

**Precision (Turkish: *kesinlik*)** — of the pixels we *predicted* would burn, what fraction actually burned?

$$\text{Precision} = \frac{TP}{TP + FP}$$

This is the "can I trust the alarm?" metric. Low precision means many false alarms.

**Recall (Turkish: *duyarlılık / anma*)**, also called sensitivity or true positive rate — of the pixels that *actually* burned, what fraction did we find?

$$\text{Recall} = \frac{TP}{TP + FN}$$

This is the "did I miss anything?" metric. Low recall means missed fires.

**They trade off against each other.** Lower the decision threshold $\tau$ and you predict fire more often: recall rises, precision falls. Raise $\tau$ and the reverse happens. At the extremes:

| Strategy | Precision | Recall |
|---|---|---|
| Predict fire everywhere ($\tau \to 0$) | prevalence = 0.0027 | 1.00 |
| Predict fire nowhere ($\tau \to 1$) | undefined | 0.00 |

Neither is useful, which is why you need a metric that combines them.

**F1-score (Turkish: *F1 skoru*)** — the **harmonic mean** of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Why the *harmonic* mean rather than the ordinary arithmetic mean? Because the harmonic mean is dominated by the smaller value. Compare a model with precision 1.00 and recall 0.01:

- Arithmetic mean: $(1.00 + 0.01)/2 = 0.505$ — looks acceptable!
- Harmonic mean: $2(1.00 \times 0.01)/(1.01) = 0.0198$ — correctly reports that the model is nearly useless.

F1 refuses to let you succeed by being good at only one of the two.

**IGNIS values (τ = 0.5, full archive):**

| Metric | Value | Reading |
|---|---|---|
| Precision | **0.0601** | Of every 100 pixels we flag as burning tomorrow, 6 actually burn. 94 are false alarms. |
| Recall | **0.0222** | Of every 100 pixels that actually burn, we find 2. We miss 98. |
| F1 | **0.0324** | The honest combined summary. |

State those interpretations in exactly those words. "6 out of 100" and "we miss 98 out of 100" are far more communicative than "precision 0.06" and are much harder to hide behind.

### 6.4 IoU

**Intersection over Union** (IoU, Turkish: *kesişimin birleşime oranı*), also known as the **Jaccard index**, is the standard metric for segmentation.

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

where $A$ is the predicted mask and $B$ is the true mask.

```
      predicted           true              IoU = overlap / total covered
     ┌────────┐                            
     │  ┌─────┼────┐                        ┌────────┐
     │  │/////│    │                        │  ┌─────┼────┐
     │  │/////│    │        →               │//│█████│////│
     └──┼─────┘    │                        └──┼─────┘////│
        └──────────┘                           └──────────┘
                                            █ = TP (intersection)
                                            all shaded = union
```

**Why segmentation prefers IoU over accuracy.** Because IoU, like precision/recall/F1, never counts TN. The enormous ocean of correctly-predicted background pixels — 91.6 million of them — contributes nothing. IoU only asks: how well do the two shapes overlap?

**Relationship to F1.** IoU and F1 (which for segmentation is the same as the Dice coefficient) are monotonically related:

$$\text{IoU} = \frac{F_1}{2 - F_1}, \qquad F_1 = \frac{2\,\text{IoU}}{1 + \text{IoU}}$$

Check with our numbers: $F_1 = 0.0324 \Rightarrow \text{IoU} = 0.0324/(2-0.0324) = 0.0324/1.9676 = 0.01647$. The reported IoU is **0.0165**. The numbers are internally consistent, which is a good sign that the evaluation code is correct.

IoU is always $\le$ F1. IoU is the harsher of the two, which is why segmentation papers report it.

**IGNIS IoU = 0.0165.** Interpretation: the predicted burn area and the true burn area overlap by 1.65 % of their combined footprint. For reference, a segmentation result is generally considered usable somewhere above IoU ≈ 0.5, and state-of-the-art natural-image segmentation reaches 0.8+. We are two orders of magnitude away from usable.

### 6.5 The ROC curve

The **ROC curve** (Receiver Operating Characteristic, Turkish: *alıcı işletim karakteristiği eğrisi*) plots, as the threshold $\tau$ sweeps from 1 down to 0:

- **y-axis: True Positive Rate** = Recall = $TP/(TP+FN)$
- **x-axis: False Positive Rate** = $FP/(FP+TN)$

```
   TPR
   1.0 ┤        ╭─────────────  perfect (AUC = 1.0)
       │      ╭─╯      ╱
       │    ╭─╯      ╱
       │  ╭─╯      ╱     ← random guessing (AUC = 0.5)
       │╭─╯      ╱
   0.0 ┼───────╱────────────────
      0.0                    1.0  FPR
```

**ROC-AUC** is the area under this curve. It has a beautiful probabilistic interpretation:

> **ROC-AUC is the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative example.**

So ROC-AUC = 0.5 means "no better than a coin flip at ranking", and 1.0 means "perfect ranking".

**IGNIS ROC-AUC = 0.8468.** Interpretation: if you pick a pixel that burns tomorrow and a pixel that does not, our model gives the burning one a higher probability **84.68 % of the time**. That is genuinely, substantially better than chance. It proves the 14-channel representation carries real information.

So why is the model useless? Look at the denominator of the FPR.

### 6.6 The PR curve and AUC-PR

The **precision–recall curve** plots, as $\tau$ sweeps:

- **y-axis: Precision** = $TP/(TP+FP)$
- **x-axis: Recall** = $TP/(TP+FN)$

**AUC-PR** (Turkish: *kesinlik-duyarlılık eğrisi altındaki alan*), also called Average Precision, is the area under it.

The decisive difference from ROC:

| | ROC curve | PR curve |
|---|---|---|
| x-axis denominator | $FP + TN$ — **includes the 91.6 million negatives** | $TP + FN$ — only the 246,730 positives |
| y-axis denominator | $TP + FN$ | $TP + FP$ — **no TN anywhere** |
| Uses TN? | Yes, heavily | **No** |
| Baseline for a random classifier | Always 0.5 | **Equals the prevalence** |

**The baseline point is critical.** A random classifier's PR curve is a horizontal line at $y = $ prevalence. For IGNIS:

$$\text{AUC-PR}_{\text{random}} = \text{prevalence} = 0.002686 \approx \mathbf{0.0027}$$

So "AUC-PR = 0.0210" is meaningless until you know that the baseline is 0.0027. Our value is

$$\frac{0.0210}{0.002686} = \mathbf{7.8\times} \text{ the prevalence baseline}$$

This is why the manuscript says the AUC-PR is *"about 7.8 times the 0.00269 prevalence baseline"* — and immediately adds that it is nevertheless *"far too low for operational use"*. Both halves of that sentence must be spoken together. Reporting only "7.8× better than random" would be spin; reporting only "0.0210" without the baseline would be uninterpretable.

### 6.7 Our own contradiction: ROC-AUC 0.8468 versus AUC-PR 0.0210

Here is the apparent paradox, and it is the single best teaching example in this entire guide because it is *our own data*.

| Metric | Value | Says |
|---|---|---|
| ROC-AUC | **0.8468** | The model is good |
| AUC-PR | **0.0210** | The model is nearly useless |

Both are correctly computed. Both are true. They are measuring different things, and under extreme imbalance they diverge violently.

**The resolution, with numbers.**

The false positive rate is $FP/(FP+TN)$. Our negative population is about **91.6 million pixels**. Suppose the model produces 100,000 false positives. Then:

$$\text{FPR} = \frac{100{,}000}{91{,}610{,}166} = 0.0011$$

On the ROC curve, that is essentially zero. The curve does not move. ROC-AUC stays high.

But now compute precision at that same operating point. Suppose those 100,000 predictions contain 6,000 true positives:

$$\text{Precision} = \frac{6{,}000}{106{,}000} = 0.057$$

**94 % of our alarms are wrong** — and the ROC curve is completely blind to it, because 100,000 false positives is a rounding error next to 91.6 million negatives.

Put more starkly:

> Under extreme imbalance, the *number of negatives is so large that the false positive rate can be tiny while the number of false positives is operationally catastrophic.* ROC-AUC measures the rate. Precision measures the count. Only one of them matters to a duty officer deciding where to send a helicopter.

**A concrete thought experiment for the jury.** Imagine deploying IGNIS over Türkiye. Türkiye's land area is about 780,000 km², i.e. 780,000 of our 1 km pixels. If the model flags 1 % of them as "will burn tomorrow", that is 7,800 alarms per day. At precision 0.0601, about 469 of those are real and about 7,331 are false. No fire service in the world can act on 7,331 false alarms per day. The model is not deployable, and ROC-AUC 0.847 does not change that.

**The rule:**

> **When the positive class is rare, use AUC-PR as the primary metric, always report the prevalence alongside it, and treat ROC-AUC as a secondary diagnostic of ranking quality only.**

The manuscript adopts exactly this position: AUC-PR is *"the primary metric because it is insensitive to the very large true-negative population"*. Say this if challenged.

**What ROC-AUC 0.847 *does* legitimately tell us.** It is not worthless. It says the model has learned a real ranking: burning pixels systematically get higher scores. That means the input features are informative and the architecture is not broken. The failure is in converting a ranking into a usable *decision*. That is an encouraging diagnosis, because ranking is the hard part and calibration is comparatively fixable.

### 6.8 Threshold selection and calibration

The model outputs a continuous probability $P(i,j) \in (0,1)$. To produce a binary mask you must pick a **threshold** (Turkish: *eşik*) $\tau$.

**IGNIS used $\tau = 0.5$, and that is almost certainly wrong.**

Why is 0.5 the default? Because $\sigma(0) = 0.5$, so it corresponds to "the raw logit is positive". It is the natural choice when the two classes are balanced. Our classes are imbalanced by a factor of **372 to 1** ($1/0.002686 = 372$). There is no reason at all to expect the optimal cut to sit at 0.5.

The consequence is visible in our own numbers: recall 0.0222 is *lower* than precision 0.0601, which means the threshold is far too conservative. The model is being asked to be very sure before it says "fire", and it almost never is. Lowering $\tau$ would trade some precision for a lot of recall, and would very likely raise F1 substantially — **without retraining anything**.

**The correct procedure**, which the new pipeline implements:

1. Train the model.
2. On the **validation set only**, compute precision and recall at many candidate thresholds (e.g. $\tau = 0.01, 0.02, \dots, 0.99$).
3. Choose the $\tau^\star$ that maximises F1 (or, if the operational cost asymmetry is known, maximises $F_\beta$ with $\beta > 1$ to favour recall).
4. Freeze $\tau^\star$ and apply it unchanged to the test set.

Step 4 is essential. If you choose the threshold on the test set, you have leaked test information into your model and your reported score is inflated.

**A related but distinct idea: calibration (Turkish: *kalibrasyon*).** A model is *calibrated* if, among all pixels it assigns probability 0.3, about 30 % actually burn. Focal loss and positively weighted losses deliberately distort probabilities in order to fight imbalance, so they produce **miscalibrated** outputs. If IGNIS ever needs to output "there is a 30 % chance this village is threatened", a calibration step such as Platt scaling or isotonic regression would be required. For threshold-based mask generation, calibration is not strictly necessary — only the ranking matters. This is a good distinction to be able to make if asked about probabilities.

### 6.9 Baselines, and why persistence is mandatory

A **baseline** (Turkish: *temel çizgi / baz çizgisi*) is a deliberately simple method that your model must beat. Without one, a number means nothing.

For next-day fire spread, three baselines are relevant, and IGNIS's new evaluation plan includes all three.

| Baseline | Turkish | Rule | Why it matters |
|---|---|---|---|
| **Persistence** | Kalıcılık | Tomorrow's mask = today's mask | The absolute floor. Costs zero. |
| **Dilated persistence** | Genişletilmiş kalıcılık | Today's mask, expanded by 1 pixel in all directions | Adds the crudest possible notion of growth |
| **Wind-directed growth** | Rüzgâr yönlü büyüme | Today's mask, shifted/expanded along the ERA5 wind vector | Adds the dominant physical driver, with no learning |

**Persistence is the most important one and it is non-negotiable.** Fires burn for days. Yesterday's burning pixel is very likely to be burning today. A prediction of "tomorrow looks exactly like today" requires no data, no model, no GPU, and no training. **If your neural network cannot beat it, your neural network has contributed nothing.**

We measured it. On a sample of 45 files containing 1,054 patches:

| Method | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| **Persistence** ("tomorrow = today") | 0.0430 | **0.0963** | **0.0595** | **0.0306** |
| **IGNIS U-Net** (τ = 0.5) | 0.0601 | 0.0222 | 0.0324 | 0.0165 |
| **Ratio (persistence / model)** | 0.72× | **4.34×** | **1.84×** | **1.85×** |

**The model loses.** Persistence achieves 1.84× the F1 and 1.85× the IoU of a 1.9-million-parameter convolutional neural network trained on 92 million labelled pixels.

The one thing the model does better is precision (0.0601 vs 0.0430) — when it does fire an alarm, it is slightly more often right. But it fires so few alarms that its recall is 4.3 times worse, and F1 punishes that correctly.

Section 9 discusses what to do about this. For now, note the intellectual honesty required to run this comparison at all. It would have been easy not to compute it. **Computing it, publishing it, and diagnosing it is the scientific contribution.**

### 6.10 Techniques for handling imbalance

There are five families of approach. IGNIS has used one and is moving to a combination.

#### (a) Resampling

**Oversampling** (Turkish: *aşırı örnekleme*) duplicates minority-class examples; **undersampling** (Turkish: *eksik örnekleme*) discards majority-class examples.

For *pixel-level* segmentation this is awkward, because you cannot duplicate a pixel without duplicating its whole patch. What you *can* do is resample at the **patch** level — for instance, oversample patches containing many positive pixels. IGNIS does something related but more elegant: **it only extracts patches centred on active fire pixels in the first place.** Every training sample is guaranteed to contain fire. A random patch over Türkiye would be almost entirely empty.

The new pipeline goes further with the **32×32 centre crop** (Section 7.5), which is a form of spatial undersampling of the background: it discards the outer ring of the patch, which contains disproportionately many negatives, and roughly quadruples positive density.

#### (b) Class weights / positively weighted BCE

Multiply the loss of positive pixels by a factor $w > 1$ so that the rare class contributes as much total gradient as the common one.

$$L = -\big[\,w \cdot y\log p + (1-y)\log(1-p)\,\big]$$

In PyTorch this is `BCEWithLogitsLoss(pos_weight=w)`. The theoretically balanced value is

$$w = \frac{\text{number of negatives}}{\text{number of positives}} = \frac{1 - 0.002686}{0.002686} \approx 371$$

In practice $w = 371$ often destabilises training and produces a model that predicts fire almost everywhere, so values in the range 10–50 are common starting points. The repository's `SPREAD_POS_WEIGHT` default is 10 — which, given a 372:1 imbalance, is arguably far too small. This is a concrete, testable hypothesis about why the model under-predicts, and it is worth mentioning as a specific planned experiment.

#### (c) Focal loss

Introduced by Lin et al. (2017) for dense object detection — a problem with exactly our structure: a few objects, an ocean of background.

$$L_{\text{focal}} = -\alpha\,(1 - p_t)^{\gamma}\,\log(p_t)$$

where $p_t$ is the model's predicted probability *for the true class* (so $p_t = p$ if $y=1$, and $p_t = 1-p$ if $y=0$), $\alpha$ balances the classes and $\gamma$ is the **focusing parameter**.

**What does $(1-p_t)^\gamma$ actually do?** It is a down-weighting factor for easy examples. Work through it with $\gamma = 2$, the value IGNIS uses:

| Situation | $p_t$ | $(1-p_t)^2$ | Effect on that pixel's loss |
|---|---|---|---|
| Easy background, confidently right | 0.99 | $0.01^2 = 0.0001$ | Loss reduced **10,000×** |
| Easy background, right | 0.90 | $0.10^2 = 0.01$ | Loss reduced **100×** |
| Uncertain | 0.50 | $0.50^2 = 0.25$ | Loss reduced 4× |
| Hard, nearly wrong | 0.10 | $0.90^2 = 0.81$ | Loss barely reduced |
| Wrong | 0.01 | $0.99^2 = 0.98$ | Loss essentially unchanged |

The vast population of trivially-correct background pixels is suppressed by four orders of magnitude, so the total gradient becomes dominated by the sparse, difficult fire pixels. The manuscript states it exactly: *"When a background pixel is already classified with high confidence, $p_t$ approaches one and the factor $(1-p_t)^\gamma$ suppresses its contribution, so the gradient is dominated by the sparse and difficult fire pixels."*

Note that $\gamma = 0$ recovers ordinary weighted cross-entropy. Larger $\gamma$ focuses harder.

IGNIS uses $\gamma = 2.0$ and $\alpha = 0.80$. The manuscript's own diagnosis is that this was insufficient: *"at a prevalence of 0.27 %, the focal loss alone is insufficient."*

#### (d) Dice loss

Dice loss attacks the problem from a completely different direction: instead of correcting a per-pixel loss, **it makes the loss itself an overlap metric.**

The Dice coefficient is the same quantity as F1:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

To use it as a loss it must be differentiable, so we use the **soft Dice**, replacing hard counts with sums of probabilities:

$$L_{\text{Dice}} = 1 - \frac{2\sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}$$

where the sums run over all pixels, $p_i$ is the predicted probability, $y_i \in \{0,1\}$ the label, and $\epsilon$ a small constant (typically 1) preventing division by zero on an empty patch.

**Why is this immune to imbalance?** Because TN never appears in the formula. A patch with 11 positive pixels and one with 500 contribute comparably, because the denominator normalises by the mask size. Dice loss directly optimises what we actually measure.

Its weakness is instability: on a patch with **zero** positive pixels — and remember that **58.9 % of our patches have zero fire pixels tomorrow** — the numerator is 0 regardless of the prediction, and the gradient becomes ill-behaved. This is why Dice is almost never used alone.

**The new IGNIS loss is a hybrid:**

$$L = 0.5 \cdot L_{\text{BCE}}(\text{pos\_weight}) + 0.5 \cdot L_{\text{SoftDice}}$$

BCE provides a stable, well-conditioned per-pixel gradient everywhere, including on empty patches; Dice pulls the optimisation towards good region overlap. This combination is now standard practice in medical and remote-sensing segmentation, for exactly this reason.

#### (e) Tversky and Focal Tversky loss

The Tversky index generalises Dice by weighting FP and FN *differently*:

$$T = \frac{TP}{TP + \alpha\,FP + \beta\,FN}, \qquad L_{\text{Tversky}} = 1 - T$$

| $\alpha$ | $\beta$ | Result |
|---|---|---|
| 0.5 | 0.5 | Exactly the Dice coefficient |
| 1.0 | 1.0 | Exactly IoU / Jaccard |
| **0.3** | **0.7** | **Penalises FN more than FP → pushes recall up** |
| 0.7 | 0.3 | Penalises FP more → pushes precision up |

**IGNIS's alternative objective uses $\alpha = 0.3$, $\beta = 0.7$.** The reasoning is operational and you should state it in those terms: *for a wildfire warning system, a missed fire (FN) is far more costly than a false alarm (FP), so we deliberately weight FN 2.33 times more heavily.* This is not a statistical trick; it is encoding the real-world cost asymmetry into the objective function. That is a strong point to make to a jury.

**Focal Tversky** additionally raises the whole thing to a power to focus on hard cases:

$$L_{\text{FocalTversky}} = (1 - T)^{1/\gamma_{\!T}}$$

#### Summary of the loss strategy

| Version | Loss | Rationale |
|---|---|---|
| Preliminary (manuscript) | Focal, $\gamma=2.0$, $\alpha=0.80$ | Standard choice for dense imbalance; **proved insufficient** |
| New pipeline (primary) | $0.5\,$BCE(pos_weight) $+\ 0.5\,$SoftDice | Stability from BCE + region overlap from Dice |
| New pipeline (alternative) | Focal Tversky, $\alpha=0.3$, $\beta=0.7$ | Explicit recall preference for operational safety |

A final and important caveat, which the diagnosis in Section 9 makes unavoidable: **no loss function can fix a broken input representation.** If elevation with a standard deviation of 515 dominates the first convolutional layer while soil moisture with a standard deviation of 0.07 is invisible, changing the loss will not help. Normalisation must come first. That is why the new pipeline changes the *data* before it changes the *objective*.
---

## 7. The IGNIS data pipeline, line by line

This section walks through what actually happens to a photon before it becomes a number inside a neural network. Everything up to the TFRecord file happens in Google Earth Engine; everything after happens locally.

```
  ┌──────────────────────── GOOGLE EARTH ENGINE ─────────────────────────┐
  │                                                                       │
  │  8 collections  →  filter to date & Türkiye  →  reproject to          │
  │                    EPSG:32635 @ 1000 m       →  temporal compositing  │
  │                                              →  unit conversion       │
  │                                              →  derive wind_speed,    │
  │                                                 humidity, slope,      │
  │                                                 aspect                │
  │                                              →  stack into one Image  │
  │                    stratifiedSample (≤150 fire pixels/day)            │
  │                    neighborhoodToArray (65×65 kernel)                 │
  │                    Export.table.toDrive(format='TFRecord')            │
  └───────────────────────────────┬───────────────────────────────────────┘
                                  │  *.tfrecord.gz  (one shard per fire day)
                                  ▼
  ┌───────────────────────── LOCAL TRAINING ─────────────────────────────┐
  │  parse TFRecord  →  length check  →  crop  →  normalise  →  encode    │
  │  →  augment  →  batch  →  U-Net  →  loss  →  Adam  →  checkpoint      │
  └───────────────────────────────────────────────────────────────────────┘
```

### 7.1 The eight source products

**Quick reference: which satellite does each product come from?**

This is the single most likely factual question from a jury, so it is worth being
able to answer it precisely — including the two entries that are *not* satellites.

| Product ID | Platform / satellite | Instrument | Gives us |
|---|---|---|---|
| `MODIS/061/MOD14A1` | **Terra** (NASA EOS AM-1, launched Dec 1999) | MODIS | Active fire mask, day t |
| `MODIS/061/MYD14A1` | **Aqua** (NASA EOS PM-1, launched May 2002) | MODIS | Active fire mask, day t |
| `MODIS/061/MOD13Q1` | **Terra** | MODIS | NDVI (vegetation) |
| `MODIS/061/MOD11A1` | **Terra** | MODIS | Land surface temperature |
| `MODIS/061/MCD12Q1` | **Terra + Aqua combined** | MODIS | IGBP land cover / fuel class |
| `USGS/SRTMGL1_003` | **Space Shuttle Endeavour**, mission STS-99 (Feb 2000) | SRTM radar interferometer | Elevation, slope, aspect |
| `UCSB-CHG/CHIRPS/DAILY` | *Blended*: geostationary satellites (GOES, Meteosat) **plus ground rain gauges** | thermal infrared + stations | Precipitation |
| `ECMWF/ERA5_LAND/DAILY_AGGR` | **Not a satellite** — ECMWF reanalysis model | assimilates many observations | Air temp, dewpoint, wind u/v, soil moisture |

So the satellites we depend on directly are **Terra** and **Aqua**, both carrying
the MODIS instrument, plus the historical **SRTM** mission flown on Space Shuttle
Endeavour in February 2000 (the terrain is static, so a 2000 survey is still
current).

Two entries are commonly misdescribed, and getting them wrong in a presentation
would be an easy thing for a specialist to catch:

- **ERA5-Land is not a satellite.** It is a *reanalysis*: a weather model run
  backwards over the past, constrained by every observation available at the time
  (satellites, weather balloons, aircraft, surface stations). It gives a physically
  consistent estimate of the atmosphere, not a measurement.
- **CHIRPS is not purely satellite either.** It blends thermal-infrared cloud-top
  temperature from geostationary satellites with rain-gauge records on the ground.

The honest one-sentence answer is: *"Terra and Aqua for everything fire and
vegetation related, SRTM for terrain, plus a reanalysis and a blended
precipitation product for the weather."*


| Variable | GEE collection | Native resolution | Role in the physics |
|---|---|---|---|
| Active fire (Terra) | `MODIS/061/MOD14A1` | 1 km / daily | Today's fire mask; source of the target |
| Active fire (Aqua) | `MODIS/061/MYD14A1` | 1 km / daily | Merged with Terra for a second daily look |
| NDVI | `MODIS/061/MOD13Q1` | 250 m / 16 days | Fuel load |
| Land surface temperature | `MODIS/061/MOD11A1` | 1 km / daily | Surface energy state, drought stress |
| Meteorology | `ECMWF/ERA5_LAND/DAILY_AGGR` | ~9 km / daily | Air temperature, dew point, wind $u$/$v$, soil moisture |
| Precipitation | `UCSB-CHG/CHIRPS/DAILY` | ~5 km / daily | Rainfall |
| Topography | `USGS/SRTMGL1_003` | 30 m / static | Elevation, slope, aspect |
| Land cover | `MODIS/061/MCD12Q1` | 500 m / yearly | Fuel type |

Study area: Türkiye, defined by the `USDOS/LSIB_SIMPLE/2017` boundary with `country_na = 'Turkey'`. Period: fire season months June–October, years 2019–2021 in the current archive, extending to 2024 and beyond in the new one.

### 7.2 Harmonisation inside GEE

**Reprojection.** Every band is put on EPSG:32635 at 1000 m. See Section 3.5 for why nearest-neighbour must be used for `fire` and `landcover` and interpolation for the continuous fields.

**Quality masking.** MODIS QA bands are used to remove cloud-contaminated and invalid observations. FireMask classes representing water, cloud and unprocessed pixels are treated as non-fire.

**Temporal compositing.** Three different strategies for three different revisit rates:

| Band | Compositing rule | Reason |
|---|---|---|
| `ndvi` | Most recent 16-day composite within the preceding 32 days | MOD13Q1 revisit is 16 days |
| `lst` | Mean of the 3 preceding daily retrievals | Fills cloud gaps in MOD11A1 |
| Meteorology, `fire` | The value for the observation date itself | Both are genuinely daily |

**Unit conversion.** Kelvin → Celsius for LST and air temperature; metres → millimetres for ERA5 precipitation; the standard MODIS scale factor 0.0001 applied to NDVI so it lies in [−1, 1].

### 7.3 The derived channels

Three of the fourteen channels are computed rather than read.

**Wind speed:**
$$\text{wind\_speed} = \sqrt{u^2 + v^2} = \texttt{hypot}(u, v)$$

**Relative humidity via the Magnus formula.** ERA5-Land supplies air temperature $T$ and dew-point temperature $T_d$ in °C, but not relative humidity. The Magnus approximation converts them:

$$\text{RH} = 100 \cdot \exp(A - B), \qquad A = \frac{17.625\,T_d}{243.04 + T_d}, \qquad B = \frac{17.625\,T}{243.04 + T}$$

clipped to [0, 100] %.

The physics: dew point is the temperature to which air must be cooled to reach saturation. If $T_d = T$ the air is saturated (RH = 100 %). The larger the gap between $T$ and $T_d$, the drier the air. The exponential terms are the Magnus approximation to the saturation vapour pressure curve, and RH is the ratio of actual to saturation vapour pressure.

Worked example: $T = 35$ °C, $T_d = 10$ °C (a typical hot dry Antalya afternoon).
$A = 17.625 \times 10 / 253.04 = 0.6965$; $B = 17.625 \times 35 / 278.04 = 2.2183$;
$\text{RH} = 100\,e^{0.6965 - 2.2183} = 100\,e^{-1.5218} = 21.8\ \%$.
That is dangerously dry — fine dead fuels will be at their driest.

**Topography:** `ee.Terrain.products()` applied to the SRTM DEM gives `elevation`, `slope` and `aspect` in one call, using the finite-difference method described in Section 2.13.

### 7.4 Patch extraction: `stratifiedSample` and `neighborhoodToArray`

**Step 1 — choosing where to sample.** For each fire day, `stratifiedSample` selects up to `MAX_POINTS_PER_DAY = 150` locations from the active-fire pixels.

**Stratified sampling** (Turkish: *tabakalı örnekleme*) means sampling separately within each class rather than uniformly at random. Here it serves two purposes: it guarantees the samples are drawn from fire pixels rather than from the country at large (a random Turkish pixel is almost never burning), and it prevents one enormous fire event from flooding the archive. Without the 150 cap, the 2021 Manavgat fire alone could contribute tens of thousands of near-identical patches and dominate the loss.

There is a hidden condition: `MIN_FIRE_PIXELS = 5`. A day with fewer than 5 active fire pixels in the whole country is skipped, because it is likely a false detection or too small to learn from.

**Step 2 — extracting the neighbourhood.** `neighborhoodToArray` is the GEE operation that turns a raster into per-pixel arrays. Given a kernel — here `ee.Kernel.square(radius=32)` — it replaces every pixel's scalar value with the **entire array of values in its neighbourhood**.

```
  A square kernel of radius 32 covers  (2 × 32 + 1) = 65 pixels per side
  → each sampled point becomes a 65 × 65 array, per band
  → 14 bands + fire_next + fire_next2 + valid → one record per point
```

So a single sampled point produces a complete multi-channel image patch, and `Export.table.toDrive(..., 'TFRecord')` writes the whole day's collection of points as one compressed shard.

**Step 3 — cropping.** 65 is odd, and the U-Net's three 2×2 poolings require a size divisible by 8. Patches are therefore cropped to **64×64** at training time. The new pipeline crops much more aggressively — see Section 7.5.

**Step 4 — validity check.** Records truncated by masking at scene edges are detected by a length check and discarded. Applying the whole procedure to 360 fire days produced **22,426 valid patches**.

> **Historical bug worth knowing about.** An earlier notebook attached a scalar property named `fire` to each sample point. That name collided with the 65×65 `fire` band and silently collapsed it to a 1×1 value, producing `Can't parse serialized Example ... Key: fire` errors and zero valid frames at training time. The fix was to drop point properties before export and to use `unmask(0, False)`. If you are ever asked "what was the hardest bug", this is a genuine and instructive answer.

### 7.5 What is new: `fire_next2`, `valid`, and the 32×32 crop

Three changes to the exported data, each responding to a measured problem.

#### `fire_next2` — the ±1 day target

**The problem, measured.** In the sampled archive, **58.9 % of patches have zero fire pixels on day $t+1$**, while an average of **12.3 pixels are burning on day $t$**. A fire does not usually extinguish itself overnight. What has usually happened is that the satellite failed to see it: cloud, thick smoke, an unfavourable overpass time, or a fire that was smouldering rather than flaming at the moment Terra or Aqua passed overhead.

In other words, **the label is measuring satellite overpass luck as much as it is measuring fire behaviour.** The network is being asked to predict something partly random, and no architecture can learn a random target.

**The fix.** Export a second target band, `fire_next2`, holding the fire mask for day $t+2$, and define the training target as their union:

$$Y = \max\big(\text{fire\_next}(t{+}1),\ \text{fire\_next2}(t{+}2)\big)$$

The question the model now answers becomes **"will this pixel show fire activity in the next 24–48 hours?"** rather than "will MODIS happen to catch it burning tomorrow?".

Two honest points to make about this change:

- **It is a scientifically defensible reformulation, not a trick.** For operational planning a 48-hour window is arguably *more* useful than a 24-hour one. And it directly attacks a measured source of label noise.
- **It changes the problem, so results before and after are not comparable.** You must say this. Any improvement in F1 after this change is partly due to an easier target, and the comparison against baselines must be recomputed under the same definition.

#### `valid` — the fake-zero mask

**The problem, measured.** Approximately **15 % of the pixels in every patch are zero — and the identical ~15 % rate appears in every single environmental band.** That coincidence is the giveaway: real physical fields do not agree on where their zeros are. Humidity, elevation and NDVI have no reason to be zero in the same places.

The cause is in the GEE code: `clip(REGION)` followed by `unmask(0)`. Clipping to the Turkish national boundary makes every pixel outside the border *masked*; `unmask(0)` then replaces every masked pixel with the number 0. Because our patches are 65 km across and most Turkish fires are near the coast, a large fraction of every coastal patch is sea, or Greek or Syrian territory — all of it filled with zeros.

**Why this is destructive.** The model has no way to distinguish these two situations:

| Pixel meaning | `humidity` value | `elevation` value | `ndvi` value |
|---|---|---|---|
| Real: bone-dry air over a sea-level burnt plain | 0 | 0 | 0 |
| Fake: outside Türkiye, no data at all | 0 | 0 | 0 |

Zero relative humidity is physically almost impossible. Zero elevation means sea level. Zero NDVI means bare rock. The network is being taught that a large, systematically located region of every patch has these bizarre properties — and worse, that region's *shape* correlates with the coastline, which correlates with where fires are. This is a textbook **spurious correlation** (Turkish: *sahte korelasyon*), and it is one plausible explanation for why validation AUC-PR peaked at epoch 7: the network found the coastline shape almost immediately and started memorising it.

**The fix.** Export an explicit `valid` band: 1 where real data exists, 0 where it was filled. Then:

1. Add `valid` as a 21st input channel, so the network is *told* which pixels are fabricated;
2. Multiply the loss by `valid`, so no gradient is ever computed from a fabricated pixel;
3. Exclude invalid pixels from the normalisation statistics, so the fake zeros do not corrupt $\mu$ and $\sigma$.

Point 3 matters more than it looks. If 15 % of `humidity` values are a fake 0, the computed mean is pulled down and the computed standard deviation is inflated by a bimodal distribution that does not exist in nature.

#### The 32×32 centre crop

**The problem, measured.** A 65×65 patch is 4,225 pixels; a 64×64 crop is 4,096. Our fires occupy at most 65 pixels and on average **12.3**. So the signal is **below 1.5 %** of the image, and typically about 0.3 %. The network spends almost all of its capacity looking at empty land 30 km from the fire, which cannot possibly influence tomorrow's front.

**The fix.** Crop to the central **32×32** window, i.e. 1,024 pixels covering 32 km × 32 km. The fire is centred by construction, so the crop keeps the fire and throws away the far periphery.

$$\frac{4{,}096}{1{,}024} = 4.0 \Rightarrow \text{roughly } 4\times \text{ the positive pixel density}$$

Secondary benefits: 4× less computation per sample, so 4× larger batches for the same memory; and a receptive field that now comfortably exceeds the patch, so every output pixel sees the entire scene.

The cost, stated honestly: we lose long-range context. If tomorrow's spread is driven by something 40 km away, we can no longer see it. Given that 24-hour fire spread rates are on the order of a few kilometres per day, 32 km of context is almost certainly ample — but this should be verified empirically, not assumed.

### 7.6 Normalisation: the single most important fix

**The problem, measured.** The current pipeline performs **no input normalisation at all**. Here are the measured statistics of the raw channels:

| Channel | Min | Max | Standard deviation |
|---|---|---|---|
| `elevation` | −4 | 4,978 | **515.44** |
| `aspect` | 0 | 359 | **107.87** |
| `landcover` | 0 | 17 | **4.25** |
| `ndvi` | — | — | **0.20** |
| `soil_moisture` | — | — | **0.07** |

Look at the ratio: $515.44 / 0.07 = 7{,}363$. Elevation varies **seven thousand times more** than soil moisture in raw units.

**Why this destroys the first layer.** Recall from Section 5.7 that the first convolution computes

$$z = \sum_{c=1}^{14}\sum_{m,n} w_{c,m,n}\, x_{c,m,n} + b$$

All weights are initialised from the same small random distribution (Glorot/He initialisation), so at the start of training every channel has comparable weights. But the *inputs* differ by four orders of magnitude. The elevation term contributes values around ±500 × w; the soil moisture term contributes values around ±0.07 × w. **The elevation and aspect channels completely dominate the sum; the twelve other channels are numerically invisible.**

The gradient with respect to $w_c$ is proportional to $x_c$, so the elevation weights also receive gradients thousands of times larger than the soil-moisture weights. Gradient descent will therefore spend its early epochs tuning the elevation and aspect pathway and effectively ignoring humidity, wind and NDVI.

This is very likely a dominant cause of the failure. **A fire spread model that mostly sees elevation and aspect is, in effect, a static terrain model — which is the susceptibility problem we explicitly said we were not solving.**

**The fix: z-score normalisation** (Turkish: *z-skoru normalleştirme*, also called standardisation).

$$x' = \frac{x - \mu}{\sigma}$$

After this transformation every channel has mean 0 and standard deviation 1, so all fourteen — soon eleven continuous ones — enter the first layer on equal footing.

**Where the statistics must come from.** $\mu$ and $\sigma$ must be computed **from the training split only**, then applied unchanged to validation and test.

Why? Because $\mu$ and $\sigma$ are learned parameters, even though they are not learned by gradient descent. If you compute them over the whole dataset, information about the test years — how hot they were, how dry, how mountainous the fires were — flows into the transformation applied to every training sample. That is **data leakage** (Section 4.4), and your test score becomes optimistic. It is a small leak compared to shuffling patches across the split, but it is a real one, and reviewers of a serious venue will ask about it.

In the new pipeline: compute $\mu, \sigma$ over the 2019–2023 training years, excluding pixels where `valid = 0`, and store them alongside the model checkpoint so that inference uses exactly the same numbers.

### 7.7 The circular variable problem: aspect

**The problem.** Aspect is measured in degrees clockwise from north, 0–359. Consider two slopes:

| Slope | Aspect | Physical direction |
|---|---|---|
| A | 359° | Almost exactly north |
| B | 1° | Almost exactly north |

They are physically **2° apart**. Numerically they are **358 apart** — a bigger difference than between north (0°) and south (180°). To a neural network, which only sees numbers, north-facing slopes are split into two populations at opposite ends of the input range.

Worse: the raw standard deviation of 107.87 tells you the channel has enormous numerical spread, so after the *lack* of normalisation described above, aspect is the second-most-dominant channel in the first layer — and its numerical structure is meaningless.

**The fix: sine/cosine encoding.** Replace the single `aspect` channel by two channels:

$$\text{aspect}_{\sin} = \sin\!\left(\frac{\pi \cdot \text{aspect}}{180}\right), \qquad \text{aspect}_{\cos} = \cos\!\left(\frac{\pi \cdot \text{aspect}}{180}\right)$$

Both lie in [−1, 1], which is already well scaled, and the discontinuity disappears:

| Aspect | Direction | sin | cos |
|---|---|---|---|
| 0° | North | 0.000 | **1.000** |
| 1° | North | 0.017 | 1.000 |
| 90° | East | **1.000** | 0.000 |
| 180° | South | 0.000 | **−1.000** |
| 270° | West | **−1.000** | 0.000 |
| 359° | North | −0.017 | 1.000 |

Compare 359° and 1°: $(-0.017, 1.000)$ versus $(0.017, 1.000)$. The Euclidean distance between them is 0.035 — genuinely small, exactly as the physics requires. Compare 0° and 180°: $(0, 1)$ versus $(0, -1)$, distance 2 — the maximum. The encoding now has the correct geometry.

An extra benefit: the two components have direct physical meaning. `aspect_cos` is a **northness** index (+1 = fully north-facing, −1 = fully south-facing), and in the northern hemisphere that is almost exactly the solar-exposure/fuel-dryness axis the network needs. `aspect_sin` is an eastness index, which interacts with morning versus afternoon heating.

The same technique applies to any circular variable: wind direction, day of year, time of day. In IGNIS, wind direction is already handled correctly, because we carry $u$ and $v$ rather than a bearing — and $(u,v)$ *is* a sine/cosine encoding scaled by speed.

### 7.8 The categorical variable problem: land cover

**The problem.** `landcover` holds the MODIS IGBP class code, an integer from 1 to 17:

| Code | IGBP class | Code | IGBP class |
|---|---|---|---|
| 1 | Evergreen needleleaf forest | 10 | Grasslands |
| 2 | Evergreen broadleaf forest | 11 | Permanent wetlands |
| 3 | Deciduous needleleaf forest | 12 | Croplands |
| 4 | Deciduous broadleaf forest | 13 | Urban and built-up |
| 5 | Mixed forest | 14 | Cropland / natural vegetation mosaic |
| 6 | Closed shrublands | 15 | Permanent snow and ice |
| 7 | Open shrublands | 16 | Barren |
| 8 | Woody savannas | 17 | Water bodies |
| 9 | Savannas | | |

These are **labels**, not quantities. But when you feed the integer 17 into a convolution, the arithmetic treats it as a magnitude. The network is implicitly told:

- Water (17) is "8.5 times" evergreen broadleaf forest (2);
- Barren (16) is "between" snow (15) and water (17);
- The average of grassland (10) and croplands (12) is permanent wetlands (11).

**Every one of those statements is nonsense.** There is no ordering on land cover classes. The numbering is arbitrary; NASA could have assigned the codes in any order and the physical world would be unchanged.

**The fix: one-hot encoding** (Turkish: *bire-bir / tek-sıcak kodlama*). Replace the single integer channel by $K$ binary channels, exactly one of which is 1:

```
   class 5 (mixed forest), K = 6 groups:

   group:      forest  shrub  grass  crop  nonfuel  wetland
   channel:  [   1  ,    0  ,   0  ,   0  ,    0   ,    0   ]
```

Now no arithmetic relationship between classes is implied. Each class gets its own independent weight in the first layer, so the network can learn that pine forest is highly flammable and water is not, without being forced to place them on a line.

**Why 6 groups rather than 17 channels?** Two reasons. First, parsimony: 17 extra channels for a variable that is constant over most of a patch is wasteful, and rare classes (permanent snow, deciduous needleleaf) would have almost no training examples in Türkiye. Second, physical relevance: what matters for fire is **fuel behaviour**, and several IGBP classes behave identically as fuel. The new pipeline therefore collapses the 17 IGBP classes into **6 fuel groups** — broadly, closed forest, shrubland/maquis, grassland and savanna, agricultural land, wetland, and non-fuel surfaces such as water, urban and barren.

> *The exact IGBP-code-to-fuel-group mapping is defined in the new pipeline's configuration and should be quoted from the code, not from memory, when writing the manuscript.*

**Channel accounting for the new 21-channel input:**

| Group | Channels | Treatment |
|---|---|---|
| Continuous | `ndvi`, `lst`, `air_temp`, `humidity`, `wind_speed`, `wind_u`, `wind_v`, `precip`, `soil_moisture`, `elevation`, `slope` | 11 channels, z-score normalised with training-split $\mu, \sigma$ |
| Circular | `aspect` | 2 channels: $\sin$, $\cos$ |
| Categorical | `landcover` | 6 channels: fuel-group one-hot |
| Binary | `fire` (today's mask) | 1 channel, already 0/1, no normalisation |
| Binary | `valid` | 1 channel, real-data mask |
| | **Total** | **21** |

Compare to the old input: 14 raw, unnormalised channels including a circular variable treated as linear and a categorical variable treated as ordinal. The number of channels went up by 50 %; the *quality* of the representation went up far more, and the parameter cost was 2,016 additional weights (Section 5.10).

### 7.9 Direction-aware data augmentation

**Data augmentation** (Turkish: *veri artırma*) creates new training samples from existing ones by applying transformations that do not change the correct answer. With only 22,426 patches from 360 days, augmentation is valuable.

The obvious transformations are horizontal and vertical flips, and 90° rotations. **But our data contains vectors, and vectors do not survive a flip unchanged.**

Consider a horizontal flip (mirror left–right, i.e. east–west):

```
   BEFORE flip                        AFTER naive flip (WRONG)
   
   wind →→→  🔥░░░                     ░░░🔥  →→→ wind
             fire spreads east          fire is on the right,
             (correct physics)          wind still blows east
                                        → fire spreading INTO the wind
                                        → physically impossible
```

Flipping the image without flipping the wind teaches the network that fires spread upwind. That is worse than no augmentation at all: it actively injects false physics.

**The correct transformation rules:**

| Transformation | Image | `wind_u` | `wind_v` | `aspect_sin` | `aspect_cos` |
|---|---|---|---|---|---|
| Horizontal flip (E–W mirror) | flip columns | **× (−1)** | unchanged | **× (−1)** | unchanged |
| Vertical flip (N–S mirror) | flip rows | unchanged | **× (−1)** | unchanged | **× (−1)** |

The reasoning:
- `wind_u` is the eastward component. Mirroring east and west reverses its sign.
- `wind_v` is the northward component. Mirroring north and south reverses its sign.
- `aspect_sin` is the eastness of the slope. An E–W mirror reverses it.
- `aspect_cos` is the northness of the slope. An N–S mirror reverses it.
- Scalars — `elevation`, `slope`, `ndvi`, `humidity`, `wind_speed` — are unchanged, because they have no direction.

**Note that 90° rotations are harder** and are best avoided unless implemented very carefully, because a 90° rotation must swap $u$ and $v$ with appropriate signs *and* rotate the aspect encoding by 90°, *and* — subtly — it changes the relationship between aspect and the sun, which is not symmetric under rotation in the northern hemisphere. Flips are the safe choice. Even the N–S flip is arguably questionable for the same solar-asymmetry reason, and an ablation testing E–W flip only would be a sensible experiment.

This is an excellent detail to raise at IAC. It shows that you understood your data physically rather than applying a standard computer-vision recipe. Most published wildfire deep-learning papers flip their patches without touching the wind channels.

### 7.10 TFRecord

**TFRecord** is TensorFlow's binary serialisation format: a sequence of length-prefixed, CRC-checked records, each holding a serialised protocol buffer. GEE can export directly to it, which is why IGNIS uses it.

Advantages: compact, streamable (you never load the whole dataset into RAM), and compressible — our shards are `.tfrecord.gz`.

Disadvantage in our context: it is a TensorFlow format, and the new pipeline is PyTorch. There are two options — read TFRecords in PyTorch through a lightweight reader library, or convert once to a native format such as `.npz`/`.npy` memory-mapped arrays or WebDataset shards. Given that the dataset is only 22,426 patches, a one-time conversion to memory-mapped NumPy arrays is likely simplest and fastest to load.

---

## 8. Training the model and GPUs

### 8.1 CPU versus GPU

| | **CPU** | **GPU** |
|---|---|---|
| Turkish | Merkezi işlem birimi | Grafik işlem birimi |
| Cores | 8–32, very complex | thousands, very simple |
| Optimised for | Latency — finish one task fast | Throughput — finish many tasks at once |
| Analogy | A few professors | Ten thousand students doing arithmetic |

A CPU core has branch prediction, out-of-order execution and large caches — it is built to run complicated, unpredictable code quickly. A GPU core has almost none of that, but there are thousands of them, and they execute the same instruction on different data simultaneously (SIMD/SIMT).

**Why does neural network training suit a GPU so perfectly?** Because at bottom, everything a neural network does is matrix multiplication, and matrix multiplication is *embarrassingly parallel*.

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

Every output element $C_{ij}$ depends only on one row of $A$ and one column of $B$. No output depends on another output. So all $M \times N$ elements can be computed **simultaneously**, by different cores, with no coordination.

A convolution is a matrix multiplication too. The standard implementation (`im2col`) unrolls every 3×3×14 receptive field into a column of a large matrix, so the whole convolution becomes one dense matrix product. This is why the same hardware built for rendering triangles turns out to be the ideal machine for deep learning.

### 8.2 CUDA versus ROCm

| | **CUDA** | **ROCm** |
|---|---|---|
| Vendor | NVIDIA | AMD |
| Full name | Compute Unified Device Architecture | Radeon Open Compute |
| Status | Proprietary, mature, dominant | Open source, newer, improving quickly |
| Kernel language | CUDA C++ | HIP (near-identical to CUDA C++) |
| DL libraries | cuDNN, cuBLAS | MIOpen, rocBLAS |

CUDA has been the de facto standard for a decade, which is why almost every tutorial you will read assumes NVIDIA hardware. **ROCm** is AMD's open-source equivalent. In PyTorch the API is deliberately identical — you still write `device = 'cuda'` and `tensor.cuda()` on a ROCm build, because the ROCm build of PyTorch maps those calls onto HIP. This surprises people, so it is worth knowing.

### 8.3 The hardware: RX 9070 XT, gfx1201, RDNA 4

| Property | Value |
|---|---|
| GPU | AMD Radeon RX 9070 XT |
| Architecture | **RDNA 4** |
| ISA target | **gfx1201** |
| Memory | 16 GB GDDR6 |
| Software stack | ROCm + PyTorch (ROCm build) |

**What is `gfx1201`?** It is the LLVM target identifier for the GPU's instruction set architecture. ROCm compiles kernels ahead of time for specific gfx targets, and a build that does not include your target simply will not run — which is why "is gfx1201 supported by this ROCm version?" is the first question to ask when something fails. RDNA 4 support arrived in relatively recent ROCm releases, so **pinning the ROCm and PyTorch versions in `requirements.txt` and recording them in the manuscript is part of reproducibility**, not an optional detail.

**Why the project moved from TensorFlow to PyTorch.** Three practical reasons worth stating:

1. **TensorFlow has no GPU support on native Windows** from version 2.11 onward — for any vendor, including NVIDIA. The warning message the team saw was accurate, not a bug.
2. **`tensorflow-rocm` lags behind ROCm releases** and its support for very new architectures such as RDNA 4 has historically been slow. PyTorch's ROCm builds track new hardware more quickly.
3. PyTorch's eager execution makes debugging a custom loss, a custom augmentation and a custom sampler far easier — and IGNIS needs all three.

TensorFlow has been removed entirely from the new pipeline.

**An honest caveat to have ready:** the U-Net is small (≈1.9 M parameters, 32×32 patches). CPU training is entirely feasible for this project — a matter of minutes per epoch. The GPU is a convenience that enables more experiments per day, not a scientific necessity. Do not overstate it.

### 8.4 Numerical precision and mixed precision

A floating-point number is stored as a sign bit, an exponent (which sets the *range*) and a mantissa (which sets the *precision*).

| Format | Bits | Sign | Exponent | Mantissa | Max magnitude | Relative precision |
|---|---|---|---|---|---|---|
| **float32** (fp32) | 32 | 1 | 8 | 23 | ~3.4 × 10³⁸ | ~7 decimal digits |
| **float16** (fp16) | 16 | 1 | 5 | 10 | ~6.5 × 10⁴ | ~3 decimal digits |
| **bfloat16** (bf16) | 16 | 1 | **8** | 7 | ~3.4 × 10³⁸ | ~2 decimal digits |

Read the exponent column carefully — it is the whole story.

**bfloat16 keeps float32's 8 exponent bits.** It therefore has exactly the same *dynamic range* as float32; it simply stores fewer significant digits. **float16 has only 5 exponent bits**, so its maximum is about 65,504 and its smallest normal number is about $6 \times 10^{-5}$.

**Why this makes bf16 more numerically stable for training.** Gradients in a deep network are often extremely small — $10^{-7}$ or below is routine. In float16 such a gradient **underflows to zero** and the parameter stops learning. Conversely, an unlucky large activation can **overflow to infinity**, and once one NaN appears it propagates through the entire network within one step. The standard workaround for fp16 is **loss scaling**: multiply the loss by a large constant before backpropagation, then divide the gradients afterwards, keeping everything inside fp16's narrow window. It works, but it is an extra moving part that can itself fail.

bfloat16 needs none of this. Any number representable in float32 is representable in bfloat16, just less precisely. For neural network training, range matters much more than precision — this is the central empirical finding behind bf16's adoption.

**Mixed precision (Turkish: *karma hassasiyet*)** means running the compute-heavy operations (convolutions, matrix products) in 16-bit while keeping a **master copy of the weights and the optimiser state in float32**. You get roughly 2× the arithmetic throughput and half the activation memory, with float32-quality parameter updates. Master weights must stay fp32 because a weight update can be many orders of magnitude smaller than the weight itself; in bf16, adding $10^{-6}$ to $1.0$ simply does nothing.

**Recommendation for IGNIS:** use bf16 mixed precision on RDNA 4, and if anything behaves strangely, fall back to full fp32 first before debugging anything else — the model is small enough that the speed cost is affordable.

### 8.5 VRAM and batch size

**VRAM** (Turkish: *video belleği*) is the GPU's own memory. Everything must fit: model parameters, optimiser state, activations, and the current batch.

| Consumer | Size | Note |
|---|---|---|
| Parameters | 1.93 M × 4 B ≈ **7.7 MB** | fp32 |
| Adam optimiser state | 2 × parameters ≈ **15.4 MB** | $m$ and $v$ per parameter |
| Gradients | ≈ **7.7 MB** | |
| Input batch (32 × 21 × 32 × 32, fp32) | ≈ **2.8 MB** | |
| Activations (all layers, batch 32) | tens of MB | dominates in larger models |

Total: well under 100 MB against 16 GB available. **VRAM is not remotely a constraint for IGNIS.** State this plainly if asked — it is another reason the GPU is convenience rather than necessity.

Activation memory is the term that scales with batch size, roughly linearly:

$$\text{activation memory} \propto \text{batch} \times H \times W \times \text{total channels across layers} \times \text{bytes}$$

The 32×32 crop reduces this by 4× compared to 64×64, which means you could quadruple the batch size for free — a useful side effect worth mentioning.

**How does batch size affect learning, not just memory?**

| Batch size | Gradient noise | Effect |
|---|---|---|
| Small (8–32) | High | Noisier updates, better escape from poor minima, often better generalisation |
| Large (256–1024) | Low | Smoother, faster per epoch, may converge to sharper minima that generalise worse |

A common heuristic is the **linear scaling rule**: if you multiply the batch size by $k$, multiply the learning rate by $k$ too, since each update now averages over $k$ times more data. IGNIS uses batch 32 with lr $10^{-3}$; if you move to batch 128, try lr $4\times10^{-3}$ — with a short warm-up, because large learning rates are unstable in the first few hundred steps.

### 8.6 The data-loading bottleneck

This is the practical issue you will actually hit.

```
   ┌─────────┐   read shard   ┌──────────┐  decompress  ┌─────────┐  augment  ┌─────┐
   │  DISK   │──────────────► │  CPU     │─────────────►│  CPU    │──────────►│ GPU │
   └─────────┘                └──────────┘              └─────────┘           └─────┘
        SLOW                      SLOW                     SLOW                FAST
```

If the CPU cannot prepare batches as fast as the GPU consumes them, the GPU idles. It is entirely possible to buy an expensive GPU and see 15 % utilisation because the bottleneck is gzip decompression on one CPU thread. Our data is stored as `.tfrecord.gz`, so every single batch requires decompression.

Standard remedies, all of which apply here:

| Technique | What it does |
|---|---|
| **Parallel workers** (`num_workers > 0`) | Several CPU processes prepare batches simultaneously |
| **Prefetching** | Batch $n+1$ is prepared while the GPU computes batch $n$ |
| **Pinned memory** (`pin_memory=True`) | Page-locked host memory enables faster and asynchronous host→device transfer |
| **Caching / one-time conversion** | Decompress once into a memory-mapped `.npy` array; thereafter reads are near-instant |
| **Doing augmentation on the GPU** | Flips and sign changes are trivial tensor operations |

For a dataset of 22,426 patches at 32×32×21 in float32, the whole archive is about $22{,}426 \times 21 \times 1024 \times 4 \approx 1.9$ GB — **small enough to load entirely into system RAM, or even into 16 GB of VRAM.** That single change would eliminate the data-loading bottleneck completely and is probably the highest-value engineering optimisation available. Mention it if someone asks about training efficiency.

---

## 9. Reading the results honestly

Everything in this section is measured. Nothing is estimated. Where a number is not yet available it is marked as such.

### 9.1 The archive

| Property | Value |
|---|---|
| Fire days (daily shards) | 360 |
| Valid patches | 22,426 |
| Patch size (exported / model input) | 65 × 65 / 64 × 64 |
| Labelled pixels | ≈ 91.9 million |
| Years covered | 2019, 2020, 2021 (June–October 2019 & 2020; June to late July 2021) |
| Positive pixel prevalence | **0.2686 %** |
| Mean fire pixels per patch on day $t$ | **12.3** |
| Patches with zero fire pixels on day $t+1$ | **58.9 %** |

### 9.2 Pixel-level results

Model: U-Net, ≈1.9 M parameters, focal loss ($\gamma = 2.0$, $\alpha = 0.80$), Adam lr $10^{-3}$, batch 32, evaluated at $\tau = 0.5$ over all 22,426 patches.

| Metric | Value | What it means in plain words |
|---|---|---|
| Positive prevalence | 0.2686 % | About 11 of every 4,096 pixels burn tomorrow |
| **ROC-AUC** | **0.8468** | Given one burning and one non-burning pixel, we rank them correctly 84.7 % of the time |
| **AUC-PR** | **0.0210** | 7.8× the 0.00269 random baseline — real information, unusable magnitude |
| **Precision** | **0.0601** | Of 100 pixels we flag, 6 burn; 94 are false alarms |
| **Recall** | **0.0222** | Of 100 pixels that burn, we find 2; we miss 98 |
| **F1** | **0.0324** | The honest combined score |
| **IoU** | **0.0165** | Predicted and true burn areas overlap by 1.65 % of their union |

Internal consistency check: $\text{IoU} = F_1/(2-F_1) = 0.0324/1.9676 = 0.0165$. ✓ The evaluation code is arithmetically sound; the numbers are simply low.

### 9.3 Patch-level classification and the 77 % trap

Confusion matrix, rows = observed class, columns = predicted class:

| Observed \ Predicted | Extinguishing | Stable | Growing | **Row total** |
|---|---|---|---|---|
| **Extinguishing** | **17,071** | 62 | 300 | 17,433 |
| **Stable** | 2,099 | **64** | 119 | 2,282 |
| **Growing** | 2,519 | 27 | **165** | 2,711 |
| **Column total** | 21,689 | 153 | 584 | **22,426** |

Derived per-class scores:

| Class | Precision | Recall | F1 | Share of archive |
|---|---|---|---|---|
| Extinguishing | 17,071 / 21,689 = 0.787 | 17,071 / 17,433 = **0.979** | 0.873 | **77.74 %** |
| Stable | 64 / 153 = 0.418 | 64 / 2,282 = **0.028** | 0.053 | 10.18 % |
| Growing | 165 / 584 = 0.283 | 165 / 2,711 = **0.061** | 0.100 | 12.09 % |
| **Overall accuracy** | | | **0.7714** | |
| **Macro-F1** | | | **0.3418** | |

Now the crucial comparison.

| Classifier | Accuracy |
|---|---|
| **Trivial: always predict "extinguishing"** | **0.7774** |
| **IGNIS U-Net + growth rule** | **0.7714** |
| **Difference** | **−0.0060 — the model is WORSE** |

**This is the "77 % accuracy" trap, and you must be able to explain it in thirty seconds.**

> Our patch classifier scores 77.14 % accuracy. That sounds respectable. But 77.74 % of the patches in our archive belong to the extinguishing class. A single line of code — `return "extinguishing"` — scores 77.74 %. Our neural network scores **0.6 percentage points below** that. The accuracy figure reflects the class distribution of the archive, not any skill of the model. This is exactly why we also report **macro-F1, which is 0.3418**, and why macro-F1 is the number to look at: it averages the F1 of all three classes equally, so the model cannot hide behind the majority class.

The confusion matrix shows precisely how it fails. Of 22,426 predictions, **21,689 — that is 96.7 % — are "extinguishing"**. The model has essentially learned the majority class and almost nothing else. Its recall on the class we most need to detect, *growing*, is **0.061**: of 2,711 genuinely growing fires it identified 165.

**Why macro-F1 is the right metric here.** Macro-F1 averages the per-class F1 scores without weighting by class size:

$$\text{macro-}F_1 = \frac{0.873 + 0.053 + 0.100}{3} = 0.3418$$

A trivial always-extinguishing classifier would score $(0.875 + 0 + 0)/3 \approx 0.29$. So our macro-F1 of 0.3418 is above that floor — a genuine but very small amount of skill. Note how much more informative that comparison is than "77 %".

### 9.4 The persistence comparison — the result that matters most

Measured on a sample of 45 shards containing 1,054 patches:

| Method | Precision | Recall | F1 | IoU | Cost |
|---|---|---|---|---|---|
| **Persistence** — "tomorrow = today" | 0.0430 | **0.0963** | **0.0595** | **0.0306** | zero |
| **IGNIS U-Net**, $\tau = 0.5$ | **0.0601** | 0.0222 | 0.0324 | 0.0165 | 1.9 M parameters, GPU, 92 M labelled pixels |

```
   F1 comparison

   Persistence  ████████████████████████  0.0595
   IGNIS U-Net  █████████████             0.0324
                └──────────────────────────────
                0                          0.06
```

**The model loses to persistence by a factor of 1.84 in F1 and 1.85 in IoU.**

This is the single most important sentence in the entire evaluation, and it must appear in the final manuscript. A model that cannot beat "tomorrow looks like today" has, by definition, added no value over doing nothing.

Two nuances worth stating alongside it, because they are true and they are not excuses:

1. **The model wins on precision** (0.0601 vs 0.0430). Its alarms are 40 % more reliable than persistence's. It simply issues far too few of them. That points at a threshold problem (Section 6.8) as well as a training problem.
2. **The threshold was never calibrated.** $\tau = 0.5$ was a default, not a choice. Recall (0.0222) being lower than precision (0.0601) is the signature of a threshold set too high. Recomputing the F1 at the validation-optimal $\tau^\star$ is a zero-cost experiment that should be done before any retraining, and it may well close part of this gap.

### 9.5 The seven diagnosed causes

Every item here was **measured**, not guessed. That is what makes this section a scientific contribution rather than an apology.

| # | Finding | Evidence | Severity |
|---|---|---|---|
| **1** | **Model loses to persistence** | F1 0.0324 vs 0.0595; IoU 0.0165 vs 0.0306 (45 files, 1,054 patches) | Critical — invalidates the claim of usefulness |
| **2** | **No input normalisation** | `elevation` σ = 515.44 (range −4 to 4,978), `aspect` σ = 107.87, `landcover` σ = 4.25 versus `soil_moisture` σ = 0.07, `ndvi` σ = 0.20. Ratio 7,363:1 | Critical — the first conv layer effectively sees only elevation and aspect |
| **3** | **≈15 % of every patch is a fake zero** | Identical ~15 % zero rate in *all* environmental bands, caused by `clip(REGION)` + `unmask(0)` for pixels outside the Turkish border | High — the model cannot distinguish "RH = 0 %" from "no data"; introduces a coastline-shaped spurious feature |
| **4** | **The target partly measures satellite overpass luck** | 58.9 % of patches have zero fire pixels on day $t+1$, while 12.3 pixels burn on average on day $t$ | High — cloud, smoke and orbital timing determine much of the label |
| **5** | **Patch far too large relative to the fire** | Fires occupy at most 65 pixels, mean 12.3, in a 4,225-pixel patch → signal below 1.5 % | Medium — most of the network's capacity is spent on irrelevant terrain |
| **6** | **The 2021 mega-fire is missing** | Archive ends 26 July 2021; the Manavgat and Marmaris fires began 28 July 2021 | High — the most important spread event in recent Turkish history is absent from the training data |
| **7** | **Evaluation was in-sample** | All 360 shards were evaluated, including training days | Critical — every number above is an **optimistic** bound |

Finding 7 deserves emphasis because it inverts the reading of the whole table. The manuscript already concedes it: *"The evaluation was executed over the complete archive of 360 shards, which includes the days used for training… The reported figures are consequently in-sample diagnostics and should be read as an optimistic bound."* And indeed the held-out numbers recorded during training were **worse**: AUC-PR 0.0368 at the best epoch, precision 0.076, recall 0.072.

So the honest statement is: **the true out-of-sample performance is worse than an F1 of 0.0324, and worse still relative to persistence.**

Finding 6 is the one that will most surprise a jury. Türkiye's defining modern wildfire event — the July–August 2021 Manavgat and Marmaris fires — is not in the training data. The archive covers June to *late July* 2021 and stops on 26 July; those fires started on 28 July. Two days. The most extreme spread behaviour available in the entire national record, and the archive stops two days short of it. Extending the archive is therefore not merely a matter of "more data" — it is a matter of including the behaviour regime the model most needs to learn.

### 9.6 What changes in the new architecture, and what each change addresses

| Change | Addresses finding | Expected mechanism |
|---|---|---|
| z-score normalisation of 11 continuous channels, statistics from the training split only | **#2** | All channels enter the first layer on comparable scales |
| `aspect` → $\sin$/$\cos$ (2 channels) | **#2** | Removes the false 358-unit discontinuity at north |
| `landcover` → 6 fuel-group one-hot | **#2** | Removes the false ordering among class codes |
| `valid` mask channel + loss masking | **#3** | The network is told which pixels are fabricated, and no gradient comes from them |
| ±1 day target, $Y = \max(\text{fire\_next}, \text{fire\_next2})$ | **#4** | Reduces label noise from missed overpasses; reframes as 24–48 h activity |
| 32 × 32 centre crop | **#5** | ≈4× positive density, 4× less computation, receptive field exceeds the patch |
| $0.5\,$BCE(pos_weight) $+ 0.5\,$SoftDice; alt. FocalTversky($\alpha{=}0.3,\beta{=}0.7$) | class imbalance | Stable gradients plus direct optimisation of region overlap; explicit recall preference |
| Year-based split: train 2019–2023, val 2024, test 2025–2026 | **#6, #7** | A genuine held-out test set; archive extended past July 2021 |
| Direction-aware augmentation (flip ⇒ negate `wind_u`/`wind_v`, adjust aspect $\sin$/$\cos$) | small archive | More samples without injecting false physics |
| Baselines: persistence, dilated persistence, wind-directed growth | **#1** | Every reported number is placed against what it must beat |
| Threshold $\tau$ calibrated on validation to maximise F1 | threshold | Removes the arbitrary $\tau = 0.5$ |
| PyTorch + ROCm, TensorFlow removed | tooling | GPU access on RDNA 4; easier custom losses, samplers and augmentation |

**Results of the new pipeline: to be filled in once training is complete.** Do not quote a number for the new model until it has been measured on the 2025–2026 test split with a validation-calibrated threshold and reported next to all three baselines.

### 9.7 Why honesty is a strength

You may feel that presenting a model that loses to persistence is a weakness. It is the opposite, for four reasons.

**1. A jury can detect an inflated number; it cannot detect an honest one.** Anyone in the IAF Earth Observation symposium who works on segmentation knows that 0.997 accuracy on a 0.27 %-prevalence problem means nothing. A team that reports it as a headline result loses all credibility in one slide. A team that says "accuracy is 99.7 % and that is meaningless, here is why, and here is AUC-PR against its prevalence baseline" gains it.

**2. The measured diagnosis *is* the contribution.** Anyone can train a U-Net. Very few teams measure the standard deviation of every input channel, discover a 7,363:1 scale ratio, correlate an identical 15 % zero rate across unrelated bands back to a single `unmask(0)` call, quantify that 58.9 % of their targets are empty, and connect all of it to a specific failure mode. That is real diagnostic work and it is what a scientific audience respects.

**3. It is what the manuscript already commits to.** The paper says the framework *"does not yet exceed the majority-class baseline"*, and the status note states: *"Bu sonuçlar olduğu gibi, herhangi bir iyileştirme yapılmadan sunulmuştur"* — these results are presented as they are, with no improvement applied. You have already made this choice. Defend it confidently rather than apologising for it.

**4. It is how science actually works.** A negative result with a correct diagnosis advances a field. An unreproducible positive result does not. The manuscript's own closing formulation is the right one, and it is worth memorising: *"an accurate account of what a system does not yet do is a prerequisite for improving it."*

### 9.8 The three sentences to have ready

If a jury member gives you thirty seconds, say this:

> *"IGNIS is a complete, reproducible, end-to-end pipeline: eight satellite and reanalysis products, harmonised in Google Earth Engine onto a 1 km national grid, producing 22,426 fire-centred training patches and a U-Net that predicts next-day spread per pixel.*
>
> *Our preliminary model does not yet work. Pixel F1 is 0.0324 against a persistence baseline of 0.0595, and patch accuracy of 77.1 % is below the 77.7 % majority-class floor. We report those numbers because they are what we measured.*
>
> *We have diagnosed seven specific causes from the archive itself — no input normalisation with a 7,363:1 channel scale ratio, 15 % fabricated zeros from a masking bug, 58.9 % empty targets from missed satellite overpasses, and an in-sample evaluation — and the rebuilt pipeline addresses each one directly."*

That is a strong, defensible, honest position, and it is a far better answer than a large number with no baseline.
---

## 10. Questions you may be asked, and how to answer

Thirty questions, in roughly the order a jury is likely to reach them. For each: what the questioner is really testing, and a defensible answer. Practise saying these out loud in English.

---

**Q1. Did you compare against a persistence baseline?**

*Testing: do you know that a fire spread model must beat "tomorrow = today"?*

Yes, and we lose. Persistence gives precision 0.0430, recall 0.0963, F1 0.0595 and IoU 0.0306. Our U-Net at $\tau = 0.5$ gives 0.0601, 0.0222, 0.0324 and 0.0165. Persistence is 1.84 times better in F1. We report this because a model that does not beat the trivial baseline has not demonstrated value. We have diagnosed seven specific causes and the rebuilt pipeline addresses each; persistence, dilated persistence and wind-directed growth are all now part of the standard evaluation table.

---

**Q2. Your patch accuracy is 77 %. Is that good?**

*Testing: do you understand the majority-class trap?*

No — it is worse than useless, and we say so in the paper. The extinguishing class is 77.74 % of the archive, so a one-line classifier that always predicts "extinguishing" scores 77.74 %. We score 77.14 %, which is 0.6 points *below* the trivial floor. That is why we report macro-F1, which is 0.3418, and why the confusion matrix shows that 96.7 % of our predictions are "extinguishing". The accuracy number reflects class prevalence, not skill.

---

**Q3. Your ROC-AUC is 0.847 but your AUC-PR is 0.021. Isn't that a contradiction?**

*Testing: do you understand metrics under extreme imbalance?*

Both are correct; they measure different things. ROC-AUC's false positive rate has 91.6 million true negatives in its denominator, so 100,000 false positives register as an FPR of 0.0011 — invisible on the ROC curve. Precision has no true negatives in it at all, so those same 100,000 false positives are catastrophic. ROC-AUC 0.847 tells us the ranking is genuinely informative — a burning pixel outranks a non-burning one 84.7 % of the time. AUC-PR 0.021 tells us that converting that ranking into a usable mask fails. Under 0.27 % prevalence, AUC-PR is the primary metric and must always be quoted against its baseline, which equals the prevalence, 0.00269. Our 0.021 is 7.8× that baseline — real information, unusable magnitude.

---

**Q4. Why 1 km resolution when VIIRS offers 375 m?**

*Testing: was the resolution choice deliberate?*

Because 1 km is the native resolution of the MODIS active fire product, which is both our predictor and our target, and we did not want to fabricate detail through upsampling. It also matches MOD11A1 LST at 1 km. That said, you are right that VIIRS at 375 m has comparable daily revisit and would be strictly better: it resolves fire fronts we cannot see. Our 1 km grid imposes a hard ceiling on achievable IoU, because a genuine one-pixel advance of a fire front is often simply invisible in a 1 km target mask. VIIRS (`NOAA/VIIRS/001/VNP14A1`) is explicitly named in the paper as future work.

---

**Q5. ERA5-Land is model output, not observation. Isn't that a problem?**

*Testing: do you understand what reanalysis is?*

Yes, it is a real limitation, and we chose it deliberately. ERA5-Land is a reanalysis: a frozen-version numerical weather model rerun over historical dates while assimilating all available observations. It is not a forecast and not a direct measurement. The alternative — interpolating sparse ground stations — is also a model, and a much cruder one, and there are no stations inside a burning forest. ERA5-Land gives us gap-free, physically consistent, globally uniform fields at ~9 km. The known weakness is that 9 km cannot represent terrain-driven winds in the Taurus Mountains, nor the fire's own convectively induced wind field, which no reanalysis can know about. Dynamical downscaling with a mesoscale model such as WRF would be the correct next step.

---

**Q6. How did you handle class imbalance?**

*Testing: depth of understanding, not just naming a technique.*

The preliminary model used focal loss with $\gamma = 2.0$ and $\alpha = 0.80$. The $(1-p_t)^\gamma$ factor suppresses the loss of confidently-correct background pixels by up to 10,000× at $p_t = 0.99$, so the gradient becomes dominated by the sparse fire pixels. We also handle imbalance structurally: every patch is centred on an active fire pixel, so no sample is entirely empty. Our own conclusion is that this was insufficient at 0.27 % prevalence. The rebuilt pipeline uses a hybrid loss, $0.5\,\text{BCE}(\text{pos\_weight}) + 0.5\,\text{SoftDice}$, with Focal Tversky ($\alpha = 0.3$, $\beta = 0.7$) as an alternative that deliberately penalises missed fires 2.33× more than false alarms; plus a 32×32 centre crop that quadruples positive density, and a validation-calibrated threshold instead of $\tau = 0.5$.

---

**Q7. How did you prevent data leakage?**

*Testing: rigour.*

At the patch level we split by **whole fire day**, never by individual patch. All 150 patches from one day share the same meteorology, the same terrain and heavily overlapping footprints; shuffling them across the split would put near-duplicates on both sides and inflate validation scores. The new pipeline is stricter still: a **year-based split** — train 2019–2023, validate 2024, test 2025–2026 — so an entire fire season is held out and the same fire event cannot appear on both sides. We also compute normalisation statistics $\mu$ and $\sigma$ from the training split only, because computing them over the whole dataset would leak test-set information into the input transformation.

---

**Q8. Your evaluation was in-sample. Doesn't that invalidate your results?**

*Testing: whether you already know your own weakest point.*

It makes them an optimistic upper bound, and we state this explicitly in Section 4.3 of the paper. The reported figures were computed over all 360 shards, including training days. The held-out numbers recorded during training were worse: AUC-PR 0.0368 at the best epoch, precision 0.076, recall 0.072. So the true out-of-sample performance is below the already-poor numbers we report. Correcting this to a strictly held-out protocol is the first item on our roadmap and is implemented by the year-based split.

---

**Q9. Why U-Net and not ConvLSTM?**

*Testing: architectural reasoning.*

U-Net was the right first choice for three reasons. It is the standard architecture for dense prediction from small datasets; its skip connections preserve the thin boundary structure of a fire front, which matters when the fire occupies 12 pixels out of 4,096; and it matches the problem as formulated — a single day of environmental state mapped to the next day's mask. A ConvLSTM would consume a *sequence* of days and could learn spread momentum, which is genuinely valuable physical information we currently discard. We did not start there because a recurrent model has more parameters, needs more data, and is much harder to debug — and, honestly, if the single-frame model cannot beat persistence, adding recurrence would only hide the underlying data problems. Spatiotemporal architectures are named in the paper as future work.

---

**Q10. Does your patch classifier beat the majority class?**

*Testing: the same trap as Q2, phrased so that "yes" is tempting.*

No. Majority class = 77.74 %, our model = 77.14 %. We are 0.6 points below. Macro-F1 is 0.3418 against roughly 0.29 for the trivial classifier, so there is a small amount of genuine skill, but the headline accuracy is not evidence of it.

---

**Q11. Can this system be used operationally?**

*Testing: honesty and judgement.*

No, not in its current state, and we do not claim otherwise — the paper describes it as "a methodological foundation and a documented baseline rather than a finished operational prediction system." Consider the arithmetic: Türkiye is roughly 780,000 km². If the model flagged even 1 % of pixels, that is 7,800 alarms per day; at precision 0.0601, about 7,300 of them would be false. No fire service can act on that. What *is* operationally ready is the pipeline: any date range, any region, regenerated from public collections in Google Earth Engine without local storage of raw imagery.

---

**Q12. Your prevalence is 0.27 %. Why not just predict a smoothed version of today's fire?**

*Testing: whether you know that this is exactly the dilated persistence baseline.*

That is precisely the dilated-persistence baseline, and it is now one of the three baselines in our evaluation plan. It is a serious competitor: plain persistence already beats our model. If dilated persistence beats it further, that is important information about how much of the achievable skill is pure geometry rather than environmental forcing — and any learned model must be shown to add value on top of it.

---

**Q13. Fifteen percent of your pixels are zero in every band. Explain.**

*Testing: whether you have inspected your own data.*

That was a bug we found by measuring per-channel statistics, and the giveaway was that the ~15 % zero rate was *identical* across unrelated bands — humidity, elevation and NDVI have no physical reason to be zero in the same places. The cause is `clip(REGION)` followed by `unmask(0)` in the Earth Engine code: pixels outside the Turkish national boundary are masked and then filled with the number zero. Because patches are 65 km across and Turkish fires cluster on the coast, a large fraction of every coastal patch is sea or foreign territory filled with zeros. The model cannot distinguish "relative humidity is 0 %" — physically almost impossible — from "no data". Worse, the fabricated region has a coastline shape that correlates with where fires occur, which is a spurious feature the network can memorise. The fix is an explicit `valid` band, added as a 21st input channel and used to mask the loss, and excluded from the normalisation statistics.

---

**Q14. Was your input data normalised?**

*Testing: the single most basic question in applied ML.*

In the preliminary model, no — and that is one of our seven diagnosed causes. We measured the raw channel statistics: `elevation` has standard deviation 515.44 over a range of −4 to 4,978 m, `aspect` has 107.87, `landcover` has 4.25, whereas `soil_moisture` has 0.07 and `ndvi` has 0.20. That is a 7,363:1 ratio. Because all first-layer weights are initialised at comparable magnitudes, the first convolution effectively sees only elevation and aspect, and their gradients are thousands of times larger. A fire *spread* model that mostly sees terrain has degenerated into a static *susceptibility* model — which is exactly the problem we said we were not solving. The rebuilt pipeline z-score normalises all eleven continuous channels with statistics from the training split only.

---

**Q15. Aspect is 0–360 degrees. How do you handle the wrap-around at north?**

*Testing: whether you think about variable types.*

The preliminary model did not, which is a second component of the normalisation problem. Aspect 359° and 1° are 2° apart physically but 358 units apart numerically, so north-facing slopes were split into two populations at opposite ends of the input range. The fix is a sine/cosine encoding: two channels, $\sin(\pi a/180)$ and $\cos(\pi a/180)$, both in [−1, 1]. Under that encoding 359° and 1° are at Euclidean distance 0.035 while 0° and 180° are at distance 2 — the correct geometry. As a bonus, $\cos$ is a direct "northness" index, which in the northern hemisphere is essentially the solar-exposure and fuel-dryness axis.

---

**Q16. Land cover is a class code from 1 to 17. Did you feed it as a number?**

*Testing: understanding of categorical variables.*

In the preliminary model, yes — and it implies nonsense: that water (17) is 8.5 times evergreen broadleaf forest (2), and that the average of grassland (10) and cropland (12) is permanent wetland (11). There is no ordering on IGBP class codes; NASA could have numbered them any way at all. The rebuilt pipeline collapses the 17 IGBP classes into 6 physically meaningful fuel groups and one-hot encodes them, so each group gets an independent weight and no arithmetic relationship between classes is implied.

---

**Q17. Fifty-nine percent of your patches have no fire tomorrow. How can a fire vanish overnight?**

*Testing: understanding of your own label noise.*

Usually it does not. What has usually happened is that the satellite failed to see it. MODIS reports *active fire* — a pixel actively flaming at the moment of overpass — not burned area. Cloud, thick smoke, an unfavourable overpass time, or a fire that was smouldering rather than flaming all produce a zero. So a substantial part of our target measures satellite overpass luck rather than fire behaviour, and no architecture can learn a partly random target. Our response is to export a second target band and define the label as $Y = \max(\text{fire}_{t+1}, \text{fire}_{t+2})$, i.e. "next 24–48 hour fire activity". That reduces label noise and is arguably more operationally useful; we are explicit that it changes the problem, so results before and after are not directly comparable and all baselines must be recomputed under the new definition.

---

**Q18. Why did you use FireMask ≥ 7, which includes low-confidence detections?**

*Testing: whether the threshold was a decision or an accident.*

Deliberately, to maximise recall on the fire class. FireMask 7, 8 and 9 are low, nominal and high confidence. Requiring 9 would purify the labels but discard most weak and early-stage fires — exactly the ones an operational system most needs. At 0.27 % prevalence we cannot afford to throw away two thirds of our positives. The cost is additional label noise from false detections, which we accept and state.

---

**Q19. Your patch is 65 km across but the fire is 12 pixels. Isn't the patch far too big?**

*Testing: whether the patch size was chosen or inherited.*

Yes — it is one of our seven diagnosed causes. A 64×64 patch is 4,096 pixels; the fire occupies on average 12.3 of them, so the signal is about 0.3 % and never more than about 1.5 %. Most of the network's capacity is spent on land 30 km from the fire that cannot influence tomorrow's front. The rebuilt pipeline crops to the central 32×32, which is 1,024 pixels, quadrupling positive density and cutting computation by 4×. The cost is loss of long-range context, but 24-hour spread is on the order of a few kilometres, so 32 km of context should be ample — we will verify that rather than assume it.

---

**Q20. Your archive ends in July 2021. Isn't the 2021 Manavgat fire missing?**

*Testing: whether you know your data's coverage.*

It is, and it is the most painful gap in the project. Our archive ends on 26 July 2021; the Manavgat and Marmaris fires began on 28 July 2021. Two days. The most extreme and most consequential spread behaviour in recent Turkish history is not in the training data. This matters more than "we have less data": the model has never seen the regime it most needs to learn. Extending the archive to 2022–2026 is our first data priority and is already reflected in the year-based split.

---

**Q21. What is your model's actual receptive field? Can it see enough context?**

*Testing: architectural understanding.*

With three encoder levels, two 3×3 convolutions per level and 2×2 max-pooling between them, the receptive field grows 5 → 13 → 29 → 61 pixels. At 1 km resolution, a bottleneck neuron therefore integrates a 61 km × 61 km region — essentially the whole 64 km patch. Architecturally the network can see the full context; the depth of 3 was sized to the patch. With the new 32×32 crop the receptive field comfortably exceeds the patch, which is exactly what you want for a segmentation task where global weather context matters.

---

**Q22. How do you know which input channels actually contribute?**

*Testing: whether you have done ablation.*

We have not yet, and it is on the roadmap — the paper names channel-ablation experiments explicitly. The method is to retrain with one channel or group removed and measure the drop in AUC-PR on the held-out split. Two ablations are especially informative for us: removing the wind channels, which should hurt badly if the model has learned directional spread; and removing elevation and aspect, which — given the 7,363:1 scale ratio — we suspect the current model relies on almost exclusively. We would rather run the ablation after normalisation is fixed, because ablating an unnormalised model would mostly measure the normalisation bug.

---

**Q23. Focal loss with $\gamma = 2$ and $\alpha = 0.8$ — why those values?**

*Testing: whether hyperparameters were tuned or copied.*

They are close to the values recommended by Lin et al. (2017), the paper that introduced focal loss for dense object detection — a problem with the same structure as ours. We did not tune them systematically, which is a limitation. Our measured conclusion is that focal loss alone is insufficient at 0.27 % prevalence, so rather than searching $\gamma$ and $\alpha$ further we changed loss family, to a BCE+Dice hybrid with Focal Tversky as an alternative, and we changed the data representation first, because no loss function can compensate for unnormalised inputs.

---

**Q24. Why is $\tau = 0.5$ your threshold?**

*Testing: whether you know 0.5 is arbitrary.*

It was a default, not a decision, and that is a mistake we have identified. 0.5 is the natural cut only when the classes are balanced; ours are imbalanced 372 to 1. The evidence that it is wrong is in our own numbers: recall (0.0222) is lower than precision (0.0601), the signature of a threshold set far too high. The correct procedure — implemented in the rebuilt pipeline — is to sweep $\tau$ on the validation set, choose the value that maximises F1, freeze it, and apply it unchanged to the test set. This is a zero-cost experiment that should be run before any retraining.

---

**Q25. Your model has 1.9 million parameters and 22,426 samples. Isn't it hopelessly over-parameterised?**

*Testing: understanding of capacity versus data.*

The naive ratio looks alarming, but it is the wrong comparison for segmentation: each sample carries 4,096 pixel-level labels, so the archive holds roughly 92 million labelled outputs. The real problem is that those pixels are not independent — 4,096 pixels from one patch share one weather field, one terrain and one fire, and up to 150 patches share one day. The effective sample size is closer to 360 fire days than to 92 million pixels. And we do observe overfitting: training AUC-PR reached 0.2375 against 0.0353 on validation, a 6.7× gap, with the validation peak at epoch 7. Our responses are more data (2022–2026), direction-aware augmentation, and — first — fixing the input representation, because a network that overfits to a coastline-shaped fake-zero pattern will overfit no matter how small you make it.

---

**Q26. Why is your model worse than persistence on recall but better on precision?**

*Testing: whether you can read your own confusion matrix.*

Because the model issues very few alarms. Persistence flags every currently-burning pixel, which gives it recall 0.0963. Our model at $\tau = 0.5$ flags so few pixels that its recall is 0.0222 — 4.3× worse — while the alarms it does issue are somewhat more reliable, precision 0.0601 versus 0.0430. F1 correctly punishes the imbalance between the two. This pattern is diagnostic: it points at a threshold that is too high and a loss that under-weights the positive class, both of which the rebuilt pipeline addresses.

---

**Q27. Why UTM zone 35N for a country that spans four zones?**

*Testing: geospatial competence.*

Because a machine-learning dataset needs one continuous grid. Zone 35N covers 24°–30° E; Türkiye spans roughly 26°–45° E and genuinely crosses zones 35 to 38. If each fire used its "own" zone, patches near a boundary would be in different coordinate systems and could not be stacked into a common tensor. We chose one national grid and accepted larger scale distortion in the east, which is acceptable because our archive is dominated by Aegean and western Mediterranean fires. If IGNIS is extended to eastern Anatolia, the correct fix is either a per-region zone or an equal-area projection centred on Türkiye.

---

**Q28. Your patch classifier uses thresholds 0.75 and 1.25. Where do those come from?**

*Testing: whether the labelling rule is principled.*

They define the growth ratio bands: $r = N_{t+1}/\max(N_t,1)$, with $r > 1.25$ growing, $0.75 \le r \le 1.25$ stable, $r < 0.75$ extinguishing. They are chosen, not derived. An earlier version used 1.15 and 0.85, and we widened the band because at those settings the stable class was so rare it was effectively unlearnable. The same rule is applied identically to the observed and predicted masks, so the two class sequences are directly comparable. We would present these as operational conventions rather than physical constants, and a sensitivity analysis over the thresholds would strengthen the paper.

---

**Q29. Why did you switch from TensorFlow to PyTorch?**

*Testing: engineering judgement.*

Three reasons. TensorFlow has had no GPU support on native Windows since version 2.11, for any vendor. `tensorflow-rocm` lags behind ROCm releases, and our hardware is an AMD RX 9070 XT — RDNA 4, gfx1201 — which is very recent; PyTorch's ROCm builds track new architectures more quickly. And PyTorch's eager execution makes it far easier to implement and debug a custom masked loss, a direction-aware augmentation and a stratified sampler, all three of which IGNIS needs. We should add that the model is small enough that CPU training is entirely viable; the GPU buys us more experiments per day, not a result we could not otherwise obtain.

---

**Q30. What is the single most important thing you would change?**

*Testing: judgement and priorities.*

Input normalisation, together with the `valid` mask. Those two are the cheapest changes and address the most fundamental problem: right now the first convolutional layer effectively sees only elevation and aspect, and about 15 % of what it sees is fabricated. Every other improvement — better loss, more data, a recurrent architecture — is built on top of the representation, and improving the roof before fixing the foundation would tell us nothing. After that, in order: a strictly held-out evaluation, threshold calibration, the 2022–2026 archive extension including the 2021 mega-fire, and only then architectural changes.

---

### Three questions to be ready for that have no comfortable answer

**Q31. If your model does not work, what have you actually contributed?**

A complete, reproducible, open pipeline from eight public Earth Observation products to a trained segmentation model, regenerable for any date range or region without local storage of raw imagery; the first such dataset built specifically for Türkiye; a two-level output design that converts a probability field into an operational statement; and a measured, quantitative diagnosis of why the naive application of a standard architecture to this problem fails. The last item is the one we would defend hardest. Many published wildfire deep-learning papers report headline accuracies without stating prevalence or comparing to persistence. We measured both, found we lose, and published it.

**Q32. Are you not embarrassed to present a negative result at an international congress?**

No. A negative result with a correct, measured diagnosis is a contribution; an unreproducible positive result is not. Every number we report is exactly what we measured, including the fact that our evaluation was in-sample and therefore optimistic. We would rather be the team that a specialist in the audience trusts than the team with the larger number.

**Q33. You are high-school students. How much of this did you actually do?**

The dataset design, the Earth Engine pipeline, the model, the evaluation, the diagnosis and the manuscript. The diagnosis in particular — measuring per-channel standard deviations, tracing an identical 15 % zero rate back to a single `unmask(0)` call, quantifying that 58.9 % of targets are empty, and computing the persistence baseline — is work we did on our own archive, and we can walk you through any of the numbers.

---

## 11. Glossary

English term | Turkish term | Explanation. Sorted alphabetically by the English term.

| English | Turkish | Explanation |
|---|---|---|
| **Ablation study** | Ablasyon çalışması / bileşen çıkarma analizi | Retraining a model with one component or input channel removed, to measure how much that component contributes. IGNIS plans channel ablations to test whether wind actually matters to the network. |
| **Accuracy** | Doğruluk | Fraction of all predictions that are correct. Dangerously misleading under class imbalance: predicting "no fire" everywhere gives 99.73 % accuracy in IGNIS. |
| **Activation function** | Aktivasyon fonksiyonu | A nonlinear function applied after a layer's weighted sum. Without it, stacked layers collapse into a single linear transformation. |
| **Active fire** | Aktif yangın | A pixel detected as actively flaming at the moment of satellite overpass. Not the same as burned area. |
| **Active sensing** | Aktif algılama | Remote sensing where the instrument emits its own signal (radar, LiDAR). SRTM used active radar. |
| **Adam** | — | An optimiser that keeps per-parameter adaptive learning rates from running estimates of gradient mean and variance. IGNIS uses Adam with lr $10^{-3}$. |
| **AdamW** | — | Adam with decoupled weight decay, the mathematically correct way to combine L2 regularisation with adaptive optimisers. |
| **Aspect** | Bakı | The compass direction a slope faces, 0–360° clockwise from north. A circular variable, which is why IGNIS encodes it as sine and cosine. |
| **Atmospheric correction** | Atmosferik düzeltme | Removing the effect of the atmosphere from a satellite measurement to recover surface reflectance. Already applied in the products IGNIS uses. |
| **AUC-PR** | Kesinlik-duyarlılık eğrisi altındaki alan | Area under the precision–recall curve. The primary metric for rare-class problems. Its random baseline equals the positive prevalence, 0.00269 for IGNIS. |
| **Augmentation (data)** | Veri artırma | Creating new training samples by label-preserving transformations. In IGNIS flips must also negate the wind components. |
| **Backpropagation** | Geri yayılım | The algorithm that computes the gradient of the loss with respect to every parameter in one backward pass, using the chain rule. |
| **Band** | Bant | One measured quantity in a raster, e.g. one wavelength interval or one derived variable. |
| **Baseline** | Temel çizgi / baz çizgisi | A deliberately simple method a model must beat before its performance means anything. IGNIS uses persistence, dilated persistence and wind-directed growth. |
| **Batch** | Yığın / küme | A group of samples processed together in one forward and backward pass. IGNIS uses 32. |
| **Batch normalisation** | Yığın normalizasyonu | Normalising activations across the batch to zero mean and unit variance, then applying a learned scale and shift. Applied after every convolution in IGNIS. It does *not* normalise the raw inputs. |
| **bfloat16** | — | A 16-bit float with float32's 8 exponent bits and only 7 mantissa bits. Same dynamic range as float32, so it needs no loss scaling — more numerically stable than float16 for training. |
| **Bilinear interpolation** | Çift doğrusal ara değerleme | Resampling by a weighted average of the four nearest source pixels. For continuous data only, never for class codes. |
| **Binary cross-entropy (BCE)** | İkili çapraz entropi | The standard loss for binary prediction, $-[y\log p + (1-y)\log(1-p)]$. Punishes confident mistakes heavily. |
| **Bottleneck** | Darboğaz | The deepest, most compressed layer of a U-Net. In IGNIS it is 8×8×256 and holds 46 % of the parameters. |
| **Brightness temperature** | Parlaklık sıcaklığı | The temperature a perfect black body would need to emit the observed radiance. The quantity MODIS fire detection works on. |
| **Calibration** | Kalibrasyon | A model is calibrated if among all pixels it assigns probability 0.3, about 30 % actually burn. Focal loss and weighted losses deliberately break calibration. |
| **Categorical variable** | Kategorik değişken | A variable whose values are labels with no ordering, e.g. land cover class. Must be one-hot encoded, not fed as an integer. |
| **Channel** | Kanal | One 2D map within a stack of maps. IGNIS input has 14 channels, becoming 21 in the new pipeline. |
| **CHIRPS** | — | Climate Hazards Group InfraRed Precipitation with Station data: a blended satellite-plus-gauge daily precipitation dataset at ~5 km. |
| **Circular variable** | Dairesel değişken | A variable where the maximum and minimum values are physically adjacent, e.g. aspect at 359° and 1°. Requires sine/cosine encoding. |
| **Class imbalance** | Sınıf dengesizliği | When one class vastly outnumbers another. In IGNIS the positive class is 0.2686 % of pixels, a ratio of 372:1. |
| **Classification** | Sınıflandırma | Assigning one label to a whole image, as opposed to one label per pixel. |
| **Client-side object** | İstemci tarafı nesne | In Google Earth Engine, an ordinary Python object living in your notebook, as opposed to a server-side `ee.*` handle. |
| **Co-registration** | Eş-kayıt / çakıştırma | Guaranteeing that pixel $(i,j)$ in every band refers to the same piece of ground. |
| **Confusion matrix** | Karışıklık / hata matrisi | A table of predicted class against observed class. For IGNIS's three patch classes it is 3×3. |
| **Convolution** | Evrişim | Sliding a small kernel over an input and computing the sum of element-wise products at each position. The core operation of a CNN. |
| **Coordinate reference system (CRS)** | Koordinat referans sistemi | A projection plus a datum, defining how coordinates map to positions on Earth. IGNIS uses EPSG:32635. |
| **Crown fire** | Tepe yangını | Fire burning through the tree canopy. The fastest and most destructive type; characteristic of Mediterranean pine forest. |
| **CUDA** | — | NVIDIA's proprietary GPU computing platform. The de facto standard, which is why most tutorials assume NVIDIA hardware. |
| **Data assimilation** | Veri özümsemesi | The process by which a reanalysis blends model forecasts with observations to estimate the past state of the atmosphere. |
| **Data leakage** | Veri sızıntısı | Any path by which information from the evaluation data influences the model, inflating the reported score. |
| **Datum** | Datum | The reference ellipsoid and its positioning, part of a CRS. IGNIS uses WGS 84, the same as GPS. |
| **Decoder** | Kod çözücü | The expanding half of a U-Net, which restores spatial resolution using transposed convolutions and skip connections. |
| **Deep learning** | Derin öğrenme | Machine learning with neural networks of many layers. |
| **Digital elevation model (DEM)** | Sayısal yükseklik modeli | A raster of ground elevation. IGNIS uses SRTM at 30 m, resampled to 1 km. |
| **Dice coefficient / Dice loss** | Dice katsayısı / Dice kaybı | An overlap measure numerically identical to F1. As a loss it directly optimises region overlap and ignores true negatives entirely. |
| **Dilated persistence** | Genişletilmiş kalıcılık | A baseline: today's fire mask expanded by one pixel in all directions. |
| **Dropout** | Seyreltme | Randomly zeroing a fraction of activations during training to prevent co-adaptation. IGNIS uses rate 0.2. |
| **Early stopping** | Erken durdurma | Halting training when a validation metric stops improving, and restoring the best weights. IGNIS stopped at epoch 25 and restored epoch 7. |
| **Electromagnetic spectrum** | Elektromanyetik tayf | The full range of wavelengths of electromagnetic radiation, from gamma rays to radio. |
| **Encoder** | Kodlayıcı | The contracting half of a U-Net, which trades spatial resolution for semantic abstraction. |
| **Epoch** | Devir | One complete pass over the training set. With 17,940 training patches and batch 32, one epoch is 561 iterations. |
| **EPSG code** | EPSG kodu | A unique integer identifying a coordinate reference system. 32635 = WGS 84 / UTM zone 35N. |
| **ERA5-Land** | — | ECMWF's land-surface reanalysis at ~9 km. Model output constrained by observations, not direct measurement. |
| **Export task** | Dışa aktarma görevi | An asynchronous Google Earth Engine job that computes a large result and writes it to Drive or Cloud Storage. |
| **F1-score** | F1 skoru | The harmonic mean of precision and recall. Dominated by whichever is smaller, so a model cannot succeed by being good at only one. |
| **False negative (FN)** | Yanlış negatif | Predicted no fire, but it burned. Operationally the most dangerous error. |
| **False positive (FP)** | Yanlış pozitif | Predicted fire, but it did not burn. A false alarm. |
| **Feature** | Öznitelik | An input variable. In IGNIS, one of the environmental channels. |
| **Fire behaviour triangle** | Yangın davranış üçgeni | Fuel, weather and topography — the three factor families that control how a fire spreads. |
| **Fire front** | Yangın cephesi | The advancing boundary of a burning area. A thin structure, which is why skip connections matter. |
| **Fire regime** | Yangın rejimi | The characteristic pattern of fire in an ecosystem: frequency, intensity, season, size and type. |
| **Fire spread** | Yangın yayılımı | Where an already-burning fire will be tomorrow. IGNIS's problem. Inherently temporal. |
| **Fire susceptibility** | Yangın duyarlılığı / riski | Where a fire is likely to start. Static, no temporal component. **Not** IGNIS's problem. |
| **Fire triangle** | Yangın üçgeni | Fuel, oxygen and heat — the three requirements for combustion. Distinct from the fire *behaviour* triangle. |
| **FireMask** | — | The MODIS band encoding pixel classification; 7 = low, 8 = nominal, 9 = high confidence fire. IGNIS accepts ≥ 7. |
| **float16 (fp16)** | — | A 16-bit float with only 5 exponent bits, maximum ≈ 65,504. Prone to underflow of small gradients; needs loss scaling. |
| **float32 (fp32)** | — | The standard 32-bit float: 8 exponent bits, 23 mantissa bits, ~7 decimal digits of precision. |
| **Focal loss** | Odak kaybı | $-\alpha(1-p_t)^\gamma \log p_t$. The $(1-p_t)^\gamma$ factor suppresses easy examples so hard, rare ones dominate the gradient. |
| **Fuel** | Yakıt | The plant material that burns. Characterised by load, continuity, arrangement and moisture. |
| **Fuel moisture** | Yakıt nemi | Water content of the fuel. Fine dead fuels equilibrate with air humidity in about an hour, which is why RH is such a strong predictor. |
| **Fully connected layer** | Tam bağlı katman | A layer where every input connects to every output. On a 64×64×14 patch this would require 57 million parameters for 1,000 neurons. |
| **Geostationary orbit** | Yer durağan yörünge | An orbit at 35,786 km whose period equals 24 h, so the satellite hovers over one longitude. High temporal, low spatial resolution. |
| **gfx1201** | — | The LLVM instruction-set target identifier for the AMD RDNA 4 GPU used in IGNIS. ROCm must be built for this target. |
| **Google Earth Engine (GEE)** | — | A cloud platform hosting petabytes of Earth Observation data with a parallel processing engine. IGNIS's entire preprocessing runs there. |
| **Gradient descent** | Gradyan inişi | Iteratively moving parameters in the direction that reduces the loss: $\theta \leftarrow \theta - \eta\nabla L$. |
| **Hyperparameter** | Hiperparametre | A setting chosen by the human rather than learned by gradient descent: learning rate, batch size, $\gamma$, $\tau$. |
| **IGBP** | — | International Geosphere–Biosphere Programme, whose 17-class land-cover scheme MODIS MCD12Q1 uses. |
| **Inference** | Çıkarım | Running a trained model on new data to produce predictions. |
| **Intersection over Union (IoU)** | Kesişimin birleşime oranı | $TP/(TP+FP+FN)$; the overlap of predicted and true masks as a fraction of their union. The standard segmentation metric. IGNIS: 0.0165. |
| **Iteration / step** | Yineleme / adım | One parameter update, using one batch. |
| **Jaccard index** | Jaccard indeksi | Another name for IoU. |
| **Kernel / filter** | Çekirdek / filtre | The small matrix of learned weights slid across the input in a convolution. IGNIS uses 3×3 kernels throughout. |
| **Label** | Etiket | The correct answer attached to a training sample. In IGNIS, the next-day fire mask. |
| **Land cover** | Arazi örtüsü | The physical material covering the ground. IGNIS uses MODIS MCD12Q1 IGBP classes as a fuel-type proxy. |
| **Land surface temperature (LST)** | Arazi yüzey sıcaklığı | The radiometric skin temperature of the ground, from thermal infrared. Different from 2 m air temperature, sometimes by 20 °C. |
| **Lazy evaluation** | Tembel değerlendirme | Building a description of a computation without executing it. How GEE avoids computing petabytes it does not need. |
| **Learning rate** | Öğrenme oranı | The step size in gradient descent. The most consequential hyperparameter. IGNIS uses $10^{-3}$ with halving on plateau. |
| **Loss function** | Kayıp fonksiyonu | A single differentiable number measuring how wrong a prediction is; what gradient descent minimises. |
| **Low Earth Orbit (LEO)** | Alçak Dünya yörüngesi | Orbits between roughly 160 and 2,000 km. Where Terra and Aqua fly. |
| **Macro-F1** | Makro-F1 | The unweighted mean of per-class F1 scores. Cannot be inflated by the majority class. IGNIS: 0.3418. |
| **Magnus formula** | Magnus formülü | The relation converting air temperature and dew point to relative humidity. Used to derive IGNIS's `humidity` channel. |
| **Majority class** | Çoğunluk sınıfı | The most common class. In IGNIS's patch task, "extinguishing" at 77.74 %. |
| **Map projection** | Harita projeksiyonu | A rule for flattening the curved Earth onto a plane. Always distorts something; the choice is what. |
| **Mask** | Maske | A binary raster marking which pixels satisfy a condition. IGNIS uses fire masks and a `valid` data mask. |
| **Max pooling** | Maksimum havuzlama | Taking the maximum of each 2×2 block, halving spatial size and doubling the downstream receptive field. |
| **Mixed precision** | Karma hassasiyet | Computing in 16-bit while keeping master weights and optimiser state in float32. Roughly 2× throughput. |
| **MODIS** | — | Moderate Resolution Imaging Spectroradiometer, 36 bands, flown on both Terra and Aqua. Source of five IGNIS products. |
| **NDVI** | Normalize edilmiş fark bitki örtüsü indeksi | $(\rho_{NIR}-\rho_{Red})/(\rho_{NIR}+\rho_{Red})$. High for healthy vegetation because chlorophyll absorbs red and leaf structure scatters NIR. |
| **Nearest neighbour resampling** | En yakın komşu ile yeniden örnekleme | Taking the value of the closest source pixel. The only correct method for class codes and binary masks. |
| **Neuron** | Nöron | The basic unit: a weighted sum of inputs plus a bias, passed through an activation function. |
| **Normalisation** | Normalleştirme | Rescaling inputs so that all channels have comparable magnitude. Absent from the preliminary IGNIS model — one of its seven diagnosed faults. |
| **Null model** | Boş model | The trivial predictor, e.g. always "no fire". In IGNIS it scores 99.73 % pixel accuracy and 77.74 % patch accuracy. |
| **One-hot encoding** | Bire-bir / tek-sıcak kodlama | Representing a categorical variable as $K$ binary channels, exactly one of which is 1. Removes false ordering among class codes. |
| **Optimiser** | Eniyileyici | The algorithm that applies gradients to parameters: SGD, momentum, Adam, AdamW. |
| **Orbit** | Yörünge | The path a satellite follows around the Earth. |
| **Overfitting** | Aşırı öğrenme / ezberleme | Good on training data, bad on unseen data. IGNIS shows a 6.7× train/validation AUC-PR gap. |
| **Overpass** | Geçiş | A satellite's passage over a location. Terra crosses the equator at ~10:30 a.m., Aqua at ~1:30 p.m. |
| **Oversampling** | Aşırı örnekleme | Duplicating minority-class examples to balance the classes. |
| **Padding** | Dolgu | Adding a border of zeros so the convolution output keeps the input size. IGNIS uses `same` padding. |
| **Parameter** | Parametre | A number inside the model that is learned. IGNIS has ≈1,931,585. |
| **Passive sensing** | Pasif algılama | Remote sensing that uses the Sun or the Earth's own emission as the energy source. Blocked by cloud and thick smoke. |
| **Patch** | Yama | A small image window extracted from a larger raster. IGNIS patches are 65×65, cropped to 64×64, and 32×32 in the new pipeline. |
| **Persistence baseline** | Kalıcılık temel çizgisi | Predicting that tomorrow's fire mask equals today's. Costs nothing, and currently beats the IGNIS model. |
| **Pixel** | Piksel | One cell of a raster. In IGNIS, 1 km × 1 km = 100 hectares. |
| **Planck's law** | Planck yasası | Describes the spectrum of thermal radiation emitted by a body at a given temperature. The physical basis of satellite fire detection. |
| **Pooling** | Havuzlama | Reducing spatial size by summarising each block. |
| **Precision** | Kesinlik | $TP/(TP+FP)$: of the pixels flagged as fire, the fraction that burned. IGNIS: 0.0601. |
| **Prevalence** | Yaygınlık / görülme oranı | The fraction of examples belonging to the positive class. IGNIS: 0.2686 % at pixel level. |
| **Pyrolysis** | Piroliz | The thermal decomposition of plant material into flammable gases before ignition. The mechanism by which preheating drives fire spread. |
| **Radiance** | Radyans / ışıma | Energy per unit area, solid angle and wavelength, in W·m⁻²·sr⁻¹·µm⁻¹. What a sensor physically measures. |
| **Radiometric correction** | Radyometrik düzeltme | Converting raw sensor counts into physical radiance. Already applied in the products IGNIS uses. |
| **Raster** | Raster / hücresel veri | A grid of pixels, each holding one value per band. |
| **Reanalysis** | Yeniden analiz | A reconstruction of past atmospheric states from a frozen numerical model assimilating historical observations. ERA5-Land is one. |
| **Recall** | Duyarlılık / anma | $TP/(TP+FN)$: of the pixels that burned, the fraction we found. IGNIS: 0.0222. |
| **Receptive field** | Alıcı alan | The region of the original input that affects one output value. IGNIS's bottleneck: 61×61 pixels = 61 km. |
| **Reflectance** | Yansıtırlık | The fraction of incoming sunlight a surface reflects, dimensionless in [0,1]. |
| **Regularisation** | Düzenlileştirme | Anything done to reduce overfitting: dropout, weight decay, batch normalisation, early stopping, augmentation. |
| **ReLU** | Düzeltilmiş doğrusal birim | $\max(0,x)$. Cheap, avoids vanishing gradients, used after every convolution in IGNIS. |
| **Reprojection** | Yeniden projeksiyonlama | Converting a raster from one coordinate reference system to another. |
| **Resampling** | Yeniden örnekleme | Estimating pixel values on a new grid after reprojection or scale change. |
| **ROC curve** | ROC eğrisi | True positive rate against false positive rate as the threshold sweeps. |
| **ROC-AUC** | ROC eğrisi altındaki alan | The probability that a random positive scores higher than a random negative. IGNIS: 0.8468 — good ranking, but blind to the false-alarm count under imbalance. |
| **ROCm** | — | AMD's open-source GPU computing platform, the counterpart of CUDA. |
| **Sample** | Örnek | One (features, label) pair. IGNIS has 22,426. |
| **Scale** | Ölçek | In GEE, the nominal ground size of a pixel in metres. IGNIS uses 1000. |
| **Semantic segmentation** | Anlamsal bölütleme | Assigning a class label to every pixel of an image. IGNIS's core formulation. |
| **Server-side object** | Sunucu tarafı nesne | In GEE, an `ee.*` handle to a computation that lives on Google's servers, not in your notebook. |
| **Sigmoid** | Sigmoit | $1/(1+e^{-z})$, mapping any real number to $(0,1)$. IGNIS's output activation, turning a score into a probability. |
| **Skip connection** | Atlama bağlantısı | Copying an encoder feature map directly to the matching decoder level. Restores the fine spatial detail pooling destroyed — vital for a thin fire front. |
| **Slope** | Eğim | Steepness of the terrain in degrees. Fire spread roughly doubles for every 10° of upslope. |
| **Soil moisture** | Toprak nemi | Volumetric water content of the soil, m³/m³. A drought indicator in IGNIS. |
| **Spatial resolution** | Uzamsal çözünürlük | The ground size of one pixel. |
| **Spectral resolution** | Spektral çözünürlük | The number and narrowness of wavelength bands a sensor measures. |
| **Spotting** | Sıçrama | New fires ignited by wind-carried embers ahead of the main front. How fires cross firebreaks. |
| **SRTM** | — | Shuttle Radar Topography Mission, 2000; produced a near-global 30 m DEM using radar interferometry. |
| **Stratified sampling** | Tabakalı örnekleme | Sampling separately within each class rather than uniformly. Prevents one large fire from dominating the archive. |
| **Stride** | Adım | How far a kernel moves between positions. IGNIS uses stride 1 for convolutions, 2 for pooling and transposed convolutions. |
| **Sun-synchronous orbit** | Güneş eş-zamanlı yörünge | A near-polar orbit that crosses each latitude at the same local solar time every day, making images comparable across days. |
| **Supervised learning** | Gözetimli öğrenme | Learning from examples where the correct answer is given. |
| **Swath** | Tarama genişliği | The width of the strip a sensor images in one pass. MODIS: 2,330 km. |
| **Temporal resolution** | Zamansal çözünürlük | How often a sensor revisits the same place. MODIS: 1–2 times per day. |
| **Tensor** | Tensör | A multi-dimensional array. IGNIS's input is a rank-3 tensor per sample, rank-4 per batch. |
| **Test set** | Test kümesi | Data used exactly once, at the end, for an unbiased performance estimate. IGNIS's new test split is 2025–2026. |
| **TFRecord** | — | TensorFlow's binary record format. Google Earth Engine can export to it directly. |
| **Threshold** | Eşik | The probability cut $\tau$ converting a probability map into a binary mask. IGNIS used 0.5; the new pipeline calibrates it on validation. |
| **Training set** | Eğitim kümesi | Data the model fits its parameters to. IGNIS's new train split is 2019–2023. |
| **Transposed convolution** | Ters / transpoze evrişim | A learnable upsampling operation used in the U-Net decoder to double spatial resolution. |
| **True negative (TN)** | Doğru negatif | Predicted no fire, and none occurred. In IGNIS there are ≈91.6 million of them, which is why metrics that count TN are useless here. |
| **True positive (TP)** | Doğru pozitif | Predicted fire, and it burned. |
| **Tversky loss** | Tversky kaybı | A generalisation of Dice that weights false positives and false negatives differently. IGNIS uses $\alpha=0.3$, $\beta=0.7$ to prioritise recall. |
| **U-Net** | — | An encoder–decoder segmentation network with skip connections, introduced by Ronneberger et al. (2015). IGNIS's architecture. |
| **Underfitting** | Yetersiz öğrenme | Poor performance on both training and validation data; the model is too simple or trained too little. |
| **Undersampling** | Eksik örnekleme | Discarding majority-class examples to balance the classes. |
| **Unmask** | Maskeyi kaldırma | The GEE operation replacing masked pixels with a fixed value. `unmask(0)` is the source of IGNIS's 15 % fake zeros. |
| **Upsampling** | Yukarı örnekleme | Increasing raster resolution. Creates no new information — ERA5's 9 km fields remain 9 km fields after upsampling to 1 km. |
| **UTM** | Evrensel Enlem Dilimi Merkatör | A projection system dividing the world into 60 six-degree zones, each with its own transverse Mercator projection. |
| **Validation set** | Doğrulama kümesi | Data used to choose hyperparameters, stop training and calibrate the threshold. IGNIS's new validation split is 2024. |
| **Vector** | Vektör | A quantity with both magnitude and direction, such as wind. Must be transformed consistently under image flips. |
| **VIIRS** | — | Visible Infrared Imaging Radiometer Suite, providing 375 m active fire detection. Named in the paper as a future upgrade from MODIS's 1 km. |
| **VRAM** | Video belleği | The GPU's dedicated memory. Not a constraint for IGNIS, whose whole model and optimiser state fit in under 100 MB. |
| **Weight** | Ağırlık | A learned multiplier on an input inside a neuron or kernel. |
| **Weight decay** | Ağırlık azaltma | Penalising large weights to reduce overfitting; equivalent to L2 regularisation. |
| **Weight sharing** | Ağırlık paylaşımı | Using the same kernel at every spatial position. The reason a convolution needs 4,064 parameters where a dense layer would need 57 million. |
| **Wind components (u, v)** | Rüzgâr bileşenleri | $u$ = eastward, $v$ = northward, in m/s. Together they encode direction; their norm is wind speed. |
| **z-score normalisation** | Z-skoru normalleştirme | $x' = (x-\mu)/\sigma$, giving each channel zero mean and unit variance. Statistics must come from the training split only. |

---

## 12. References and further reading

The numbering follows the manuscript so that you can move between the two documents without confusion.

### Core methodological references

**[10] Huot, F., Hu, R.L., Goyal, N., Sankar, T., Ihme, M., Chen, Y.-F. (2022). *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data.* IEEE Transactions on Geoscience and Remote Sensing 60, 1–13.**
*Why read it:* This is the paper IGNIS is modelled on. It defines exactly our problem — daily remote-sensing predictors paired with the following day's fire mask — for the contiguous United States, and it establishes that the task is learnable and hard. Read it for the dataset design, the channel selection, and above all for how they report metrics under extreme imbalance. If a jury member asks "who else has done this?", this is the first name to say.

**[17] Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015, LNCS 9351, Springer, 234–241.**
*Why read it:* The origin of our architecture. It is short, clear, and the figure of the U-shaped network is the one every subsequent paper redraws. Read Section 2 for the skip-connection argument — it is written for a problem with the same structure as ours: few labelled images, thin structures that must be delineated precisely.

**[18] Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017, 2980–2988.**
*Why read it:* Our loss function. The paper's opening argument — that in dense prediction the background overwhelms the objective — is exactly our problem stated in a different domain. Section 3 gives the $(1-p_t)^\gamma$ derivation and the empirical study of $\gamma$ and $\alpha$ that our values come from.

**[19] Kingma, D.P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015.**
*Why read it:* Our optimiser. Read Algorithm 1 and Section 2 to understand what the first and second moment estimates do and why bias correction is needed in the first few steps. Being able to say what $\beta_1$ and $\beta_2$ mean is a small thing that signals competence.

### Data and platform references

**[12] Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., Moore, R. (2017). *Google Earth Engine: Planetary-Scale Geospatial Analysis for Everyone.* Remote Sensing of Environment 202, 18–27.**
*Why read it:* The platform on which the entire IGNIS preprocessing chain runs. Read it for the architecture section — lazy evaluation, the distributed execution model, and why the client/server distinction exists. It explains why `getInfo()` in a loop is a mistake.

**[13] Giglio, L., Schroeder, W., Justice, C.O. (2016). *The Collection 6 MODIS Active Fire Detection Algorithm and Fire Products.* Remote Sensing of Environment 178, 31–41.**
*Why read it:* This is where our target variable comes from. Read the contextual-test description and the false-alarm rejection tests. It is also the honest source for what MODIS *cannot* detect, which is the basis of our label-noise diagnosis: 58.9 % of our targets are empty, and this paper explains why.

**[14] Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., et al. (2021). *ERA5-Land: A State-of-the-Art Global Reanalysis Dataset for Land Applications.* Earth System Science Data 13(9), 4349–4383.**
*Why read it:* Our meteorology. Read the production section to understand what "reanalysis" means and the evaluation section for the known biases — you will be asked about this. It is the reference that lets you answer Q5 with authority.

**[15] Funk, C., Peterson, P., Landsfeld, M., et al. (2015). *The Climate Hazards Infrared Precipitation with Stations — A New Environmental Record for Monitoring Extremes.* Scientific Data 2, 150066.**
*Why read it:* Our precipitation. Read it for how satellite cold-cloud-duration estimates are blended with gauge data, which is the justification for preferring CHIRPS over ERA5 rainfall.

**[16] Farr, T.G., Rosen, P.A., Caro, E., et al. (2007). *The Shuttle Radar Topography Mission.* Reviews of Geophysics 45(2), RG2004.**
*Why read it:* Our topography. Read the mission description for how radar interferometry produces elevation — it is a genuinely elegant piece of engineering, and it is a good example of *active* remote sensing to contrast with all our passive products.

### Wildfire science and review references

**[6] Jain, P., Coogan, S.C.P., Subramanian, S.G., Crowley, M., Taylor, S., Flannigan, M.D. (2020). *A Review of Machine Learning Applications in Wildfire Science and Management.* Environmental Reviews 28(4), 478–505.**
*Why read it:* The best single orientation to the field. It surveys several hundred studies and organises them by problem type — which is where the susceptibility-versus-spread distinction that defines IGNIS comes from. Read it first if you read only one review.

**[8] Andrianarivony, H.S., Akhloufi, M.A. (2024). *Machine Learning and Deep Learning for Wildfire Spread Prediction: A Review.* Fire 7(12), 482.**
*Why read it:* The most directly relevant review, and the one to cite when defending your metric choices. It makes precisely our argument: metric choice and class imbalance dominate reported performance, and results are not comparable unless prevalence is stated. If someone compares your numbers to another paper's, this reference is your ground.

**Rothermel, R.C. (1972). *A Mathematical Model for Predicting Fire Spread in Wildland Fuels.* USDA Forest Service Research Paper INT-115.**
*Why read it:* The classical physical alternative to machine learning, and still the basis of operational fire-behaviour software worldwide. Read it for the slope and wind factors — it is where the physics in Section 1.3 and 1.4 of this guide comes from. Being able to say "our data-driven approach complements Rothermel-type physical models, and physics-informed hybrids are an active research direction" is a strong answer.

**[9] Pham, B.T., Jaafari, A., Avand, M., et al. (2020). *Performance Evaluation of Machine Learning Methods for Forest Fire Modeling and Prediction.* Symmetry 12(6), 1022.**
*Why read it:* This is the source of the "ROC-AUC above 0.93" figure for *susceptibility*. Read it specifically so that you can explain, with a citation, why those numbers are not comparable to ours.

**[11] Shadrin, D., Illarionova, S., Gubanov, F., et al. (2024). *Wildfire Spreading Prediction Using Multimodal Data and Deep Neural Network Approach.* Scientific Reports 14, 2606.**
*Why read it:* A recent multimodal deep-learning treatment of the spread problem; useful for positioning IGNIS in the current literature and for comparing channel selections.

**[7] Alkhatib, R., Sahwan, W., Alkhatieb, A., Schütt, B. (2023). *A Brief Review of Machine Learning Algorithms in Forest Fires Science.* Applied Sciences 13(14), 8275.**
*Why read it:* A compact survey of algorithm families. Useful for the Related Work table in the manuscript.

### Context and motivation references

**[1] Reid, C.E., Brauer, M., Johnston, F.H., Jerrett, M., Balmes, J.R., Elliott, C.T. (2016). *Critical Review of Health Impacts of Wildfire Smoke Exposure.* Environmental Health Perspectives 124(9), 1334–1343.**
*Why read it:* The human-health justification in the introduction. Gives you concrete evidence for the claim that wildfire smoke causes measurable respiratory, cardiovascular and perinatal harm.

**[2] Gill, A.M., Stephens, S.L., Cary, G.J. (2013). *The Worldwide "Wildfire" Problem.* Ecological Applications 23(2), 438–454.**
*Why read it:* Establishes that the problem is global rather than regional — useful framing for an international audience.

**[3] Elhami-Khorasani, N., Ebrahimian, H., Buja, L., et al. (2022). *Conceptualizing a Probabilistic Risk and Loss Assessment Framework for Wildfires.* Natural Hazards 114, 1153–1169.**
**[4] Carvalho, A., Monteiro, A., Flannigan, M., Solman, S., Miranda, A.I., Borrego, C. (2011). *Forest Fires in a Changing Climate and Their Impacts on Air Quality.* Atmospheric Environment 45(31), 5545–5553.**
*Why read them:* The climate-change trend argument: longer fire seasons, more intense events, and downstream air-quality consequences.

**[5] Bailon-Ruiz, R., Bit-Monnot, A., Lacroix, S. (2022). *Real-Time Wildfire Monitoring with a Fleet of UAVs.* Robotics and Autonomous Systems 152, 104071.**
*Why read it:* The complementary observation strategy. Useful for explaining why satellite EO and UAVs answer different operational questions — minutes-to-hours tactical versus daily national-scale.

### Suggested reading order for a student starting from zero

1. Jain et al. 2020 [6] — get oriented in the field.
2. Huot et al. 2022 [10] — understand the exact problem IGNIS solves.
3. Ronneberger et al. 2015 [17] — understand the architecture.
4. Lin et al. 2017 [18] — understand class imbalance and the loss.
5. Giglio et al. 2016 [13] — understand where the labels come from and what they miss.
6. Andrianarivony & Akhloufi 2024 [8] — understand how to report results honestly.

---

*This guide documents the state of IGNIS as of the preliminary baseline reported in manuscript IAC-26,B1,IP,107,x110901. Results of the rebuilt PyTorch pipeline — 21 normalised channels, ±1 day target, 32×32 crop, hybrid BCE+Dice loss, year-based split and calibrated threshold, evaluated against persistence, dilated persistence and wind-directed growth baselines — are **to be filled in once training is complete**. Do not quote a number for the new model until it has been measured on the held-out 2025–2026 test split.*
