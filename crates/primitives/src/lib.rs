//! sp1-primitives contains types and functions that are used in both sp1-core and sp1-zkvm.
//! Because it is imported in the zkvm entrypoint, it should be kept minimal.

use lazy_static::lazy_static;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::PrimeCharacteristicRing;
use p3_poseidon2::ExternalLayerConstants;
pub mod consts;
pub mod io;
pub mod types;

lazy_static! {
    // These constants are created by a RNG.

    // This will be compatible with a poseidon2 permutation config with
    // a state width of 16 and total rounds (both full and partial) of 30.
    pub static ref RC_16_30: [[BabyBear; 16]; 21] = [
        [
            BabyBear::from_u32(2110014213),
            BabyBear::from_u32(3964964605),
            BabyBear::from_u32(2190662774),
            BabyBear::from_u32(2732996483),
            BabyBear::from_u32(640767983),
            BabyBear::from_u32(3403899136),
            BabyBear::from_u32(1716033721),
            BabyBear::from_u32(1606702601),
            BabyBear::from_u32(3759873288),
            BabyBear::from_u32(1466015491),
            BabyBear::from_u32(1498308946),
            BabyBear::from_u32(2844375094),
            BabyBear::from_u32(3042463841),
            BabyBear::from_u32(1969905919),
            BabyBear::from_u32(4109944726),
            BabyBear::from_u32(3925048366),
        ],
        [
            BabyBear::from_u32(3706859504),
            BabyBear::from_u32(759122502),
            BabyBear::from_u32(3167665446),
            BabyBear::from_u32(1131812921),
            BabyBear::from_u32(1080754908),
            BabyBear::from_u32(4080114493),
            BabyBear::from_u32(893583089),
            BabyBear::from_u32(2019677373),
            BabyBear::from_u32(3128604556),
            BabyBear::from_u32(580640471),
            BabyBear::from_u32(3277620260),
            BabyBear::from_u32(842931656),
            BabyBear::from_u32(548879852),
            BabyBear::from_u32(3608554714),
            BabyBear::from_u32(3575647916),
            BabyBear::from_u32(81826002),
        ],
        [
            BabyBear::from_u32(4289086263),
            BabyBear::from_u32(1563933798),
            BabyBear::from_u32(1440025885),
            BabyBear::from_u32(184445025),
            BabyBear::from_u32(2598651360),
            BabyBear::from_u32(1396647410),
            BabyBear::from_u32(1575877922),
            BabyBear::from_u32(3303853401),
            BabyBear::from_u32(137125468),
            BabyBear::from_u32(765010148),
            BabyBear::from_u32(633675867),
            BabyBear::from_u32(2037803363),
            BabyBear::from_u32(2573389828),
            BabyBear::from_u32(1895729703),
            BabyBear::from_u32(541515871),
            BabyBear::from_u32(1783382863),
        ],
        [
            BabyBear::from_u32(2641856484),
            BabyBear::from_u32(3035743342),
            BabyBear::from_u32(3672796326),
            BabyBear::from_u32(245668751),
            BabyBear::from_u32(2025460432),
            BabyBear::from_u32(201609705),
            BabyBear::from_u32(286217151),
            BabyBear::from_u32(4093475563),
            BabyBear::from_u32(2519572182),
            BabyBear::from_u32(3080699870),
            BabyBear::from_u32(2762001832),
            BabyBear::from_u32(1244250808),
            BabyBear::from_u32(606038199),
            BabyBear::from_u32(3182740831),
            BabyBear::from_u32(73007766),
            BabyBear::from_u32(2572204153),
        ],
        [
            BabyBear::from_u32(1196780786),
            BabyBear::from_u32(3447394443),
            BabyBear::from_u32(747167305),
            BabyBear::from_u32(2968073607),
            BabyBear::from_u32(1053214930),
            BabyBear::from_u32(1074411832),
            BabyBear::from_u32(4016794508),
            BabyBear::from_u32(1570312929),
            BabyBear::from_u32(113576933),
            BabyBear::from_u32(4042581186),
            BabyBear::from_u32(3634515733),
            BabyBear::from_u32(1032701597),
            BabyBear::from_u32(2364839308),
            BabyBear::from_u32(3840286918),
            BabyBear::from_u32(888378655),
            BabyBear::from_u32(2520191583),
        ],
        [
            BabyBear::from_u32(36046858),
            BabyBear::from_u32(2927525953),
            BabyBear::from_u32(3912129105),
            BabyBear::from_u32(4004832531),
            BabyBear::from_u32(193772436),
            BabyBear::from_u32(1590247392),
            BabyBear::from_u32(4125818172),
            BabyBear::from_u32(2516251696),
            BabyBear::from_u32(4050945750),
            BabyBear::from_u32(269498914),
            BabyBear::from_u32(1973292656),
            BabyBear::from_u32(891403491),
            BabyBear::from_u32(1845429189),
            BabyBear::from_u32(2611996363),
            BabyBear::from_u32(2310542653),
            BabyBear::from_u32(4071195740),
        ],
        [
            BabyBear::from_u32(3505307391),
            BabyBear::from_u32(786445290),
            BabyBear::from_u32(3815313971),
            BabyBear::from_u32(1111591756),
            BabyBear::from_u32(4233279834),
            BabyBear::from_u32(2775453034),
            BabyBear::from_u32(1991257625),
            BabyBear::from_u32(2940505809),
            BabyBear::from_u32(2751316206),
            BabyBear::from_u32(1028870679),
            BabyBear::from_u32(1282466273),
            BabyBear::from_u32(1059053371),
            BabyBear::from_u32(834521354),
            BabyBear::from_u32(138721483),
            BabyBear::from_u32(3100410803),
            BabyBear::from_u32(3843128331),
        ],
        [
            BabyBear::from_u32(3878220780),
            BabyBear::from_u32(4058162439),
            BabyBear::from_u32(1478942487),
            BabyBear::from_u32(799012923),
            BabyBear::from_u32(496734827),
            BabyBear::from_u32(3521261236),
            BabyBear::from_u32(755421082),
            BabyBear::from_u32(1361409515),
            BabyBear::from_u32(392099473),
            BabyBear::from_u32(3178453393),
            BabyBear::from_u32(4068463721),
            BabyBear::from_u32(7935614),
            BabyBear::from_u32(4140885645),
            BabyBear::from_u32(2150748066),
            BabyBear::from_u32(1685210312),
            BabyBear::from_u32(3852983224),
        ],
        [
            BabyBear::from_u32(2896943075),
            BabyBear::from_u32(3087590927),
            BabyBear::from_u32(992175959),
            BabyBear::from_u32(970216228),
            BabyBear::from_u32(3473630090),
            BabyBear::from_u32(3899670400),
            BabyBear::from_u32(3603388822),
            BabyBear::from_u32(2633488197),
            BabyBear::from_u32(2479406964),
            BabyBear::from_u32(2420952999),
            BabyBear::from_u32(1852516800),
            BabyBear::from_u32(4253075697),
            BabyBear::from_u32(979699862),
            BabyBear::from_u32(1163403191),
            BabyBear::from_u32(1608599874),
            BabyBear::from_u32(3056104448),
        ],
        [
            BabyBear::from_u32(3779109343),
            BabyBear::from_u32(536205958),
            BabyBear::from_u32(4183458361),
            BabyBear::from_u32(1649720295),
            BabyBear::from_u32(1444912244),
            BabyBear::from_u32(3122230878),
            BabyBear::from_u32(384301396),
            BabyBear::from_u32(4228198516),
            BabyBear::from_u32(1662916865),
            BabyBear::from_u32(4082161114),
            BabyBear::from_u32(2121897314),
            BabyBear::from_u32(1706239958),
            BabyBear::from_u32(4166959388),
            BabyBear::from_u32(1626054781),
            BabyBear::from_u32(3005858978),
            BabyBear::from_u32(1431907253),
        ],
        [
            BabyBear::from_u32(1418914503),
            BabyBear::from_u32(1365856753),
            BabyBear::from_u32(3942715745),
            BabyBear::from_u32(1429155552),
            BabyBear::from_u32(3545642795),
            BabyBear::from_u32(3772474257),
            BabyBear::from_u32(1621094396),
            BabyBear::from_u32(2154399145),
            BabyBear::from_u32(826697382),
            BabyBear::from_u32(1700781391),
            BabyBear::from_u32(3539164324),
            BabyBear::from_u32(652815039),
            BabyBear::from_u32(442484755),
            BabyBear::from_u32(2055299391),
            BabyBear::from_u32(1064289978),
            BabyBear::from_u32(1152335780),
        ],
        [
            BabyBear::from_u32(3417648695),
            BabyBear::from_u32(186040114),
            BabyBear::from_u32(3475580573),
            BabyBear::from_u32(2113941250),
            BabyBear::from_u32(1779573826),
            BabyBear::from_u32(1573808590),
            BabyBear::from_u32(3235694804),
            BabyBear::from_u32(2922195281),
            BabyBear::from_u32(1119462702),
            BabyBear::from_u32(3688305521),
            BabyBear::from_u32(1849567013),
            BabyBear::from_u32(667446787),
            BabyBear::from_u32(753897224),
            BabyBear::from_u32(1896396780),
            BabyBear::from_u32(3143026334),
            BabyBear::from_u32(3829603876),
        ],
        [
            BabyBear::from_u32(859661334),
            BabyBear::from_u32(3898844357),
            BabyBear::from_u32(180258337),
            BabyBear::from_u32(2321867017),
            BabyBear::from_u32(3599002504),
            BabyBear::from_u32(2886782421),
            BabyBear::from_u32(3038299378),
            BabyBear::from_u32(1035366250),
            BabyBear::from_u32(2038912197),
            BabyBear::from_u32(2920174523),
            BabyBear::from_u32(1277696101),
            BabyBear::from_u32(2785700290),
            BabyBear::from_u32(3806504335),
            BabyBear::from_u32(3518858933),
            BabyBear::from_u32(654843672),
            BabyBear::from_u32(2127120275),
        ],
        [
            BabyBear::from_u32(1548195514),
            BabyBear::from_u32(2378056027),
            BabyBear::from_u32(390914568),
            BabyBear::from_u32(1472049779),
            BabyBear::from_u32(1552596765),
            BabyBear::from_u32(1905886441),
            BabyBear::from_u32(1611959354),
            BabyBear::from_u32(3653263304),
            BabyBear::from_u32(3423946386),
            BabyBear::from_u32(340857935),
            BabyBear::from_u32(2208879480),
            BabyBear::from_u32(139364268),
            BabyBear::from_u32(3447281773),
            BabyBear::from_u32(3777813707),
            BabyBear::from_u32(55640413),
            BabyBear::from_u32(4101901741),
        ],
        [
            BabyBear::from_u32(104929687),
            BabyBear::from_u32(1459980974),
            BabyBear::from_u32(1831234737),
            BabyBear::from_u32(457139004),
            BabyBear::from_u32(2581487628),
            BabyBear::from_u32(2112044563),
            BabyBear::from_u32(3567013861),
            BabyBear::from_u32(2792004347),
            BabyBear::from_u32(576325418),
            BabyBear::from_u32(41126132),
            BabyBear::from_u32(2713562324),
            BabyBear::from_u32(151213722),
            BabyBear::from_u32(2891185935),
            BabyBear::from_u32(546846420),
            BabyBear::from_u32(2939794919),
            BabyBear::from_u32(2543469905)
        ],
        [
            BabyBear::from_u32(2191909784),
            BabyBear::from_u32(3315138460),
            BabyBear::from_u32(530414574),
            BabyBear::from_u32(1242280418),
            BabyBear::from_u32(1211740715),
            BabyBear::from_u32(3993672165),
            BabyBear::from_u32(2505083323),
            BabyBear::from_u32(3845798801),
            BabyBear::from_u32(538768466),
            BabyBear::from_u32(2063567560),
            BabyBear::from_u32(3366148274),
            BabyBear::from_u32(1449831887),
            BabyBear::from_u32(2408012466),
            BabyBear::from_u32(294726285),
            BabyBear::from_u32(3943435493),
            BabyBear::from_u32(924016661),
        ],
        [
            BabyBear::from_u32(3633138367),
            BabyBear::from_u32(3222789372),
            BabyBear::from_u32(809116305),
            BabyBear::from_u32(30100013),
            BabyBear::from_u32(2655172876),
            BabyBear::from_u32(2564247117),
            BabyBear::from_u32(2478649732),
            BabyBear::from_u32(4113689151),
            BabyBear::from_u32(4120146082),
            BabyBear::from_u32(2512308515),
            BabyBear::from_u32(650406041),
            BabyBear::from_u32(4240012393),
            BabyBear::from_u32(2683508708),
            BabyBear::from_u32(951073977),
            BabyBear::from_u32(3460081988),
            BabyBear::from_u32(339124269),
        ],
        [
            BabyBear::from_u32(130182653),
            BabyBear::from_u32(2755946749),
            BabyBear::from_u32(542600513),
            BabyBear::from_u32(2816103022),
            BabyBear::from_u32(1931786340),
            BabyBear::from_u32(2044470840),
            BabyBear::from_u32(1709908013),
            BabyBear::from_u32(2938369043),
            BabyBear::from_u32(3640399693),
            BabyBear::from_u32(1374470239),
            BabyBear::from_u32(2191149676),
            BabyBear::from_u32(2637495682),
            BabyBear::from_u32(4236394040),
            BabyBear::from_u32(2289358846),
            BabyBear::from_u32(3833368530),
            BabyBear::from_u32(974546524),
        ],
        [
            BabyBear::from_u32(3306659113),
            BabyBear::from_u32(2234814261),
            BabyBear::from_u32(1188782305),
            BabyBear::from_u32(223782844),
            BabyBear::from_u32(2248980567),
            BabyBear::from_u32(2309786141),
            BabyBear::from_u32(2023401627),
            BabyBear::from_u32(3278877413),
            BabyBear::from_u32(2022138149),
            BabyBear::from_u32(575851471),
            BabyBear::from_u32(1612560780),
            BabyBear::from_u32(3926656936),
            BabyBear::from_u32(3318548977),
            BabyBear::from_u32(2591863678),
            BabyBear::from_u32(188109355),
            BabyBear::from_u32(4217723909),
        ],
        [
            BabyBear::from_u32(1564209905),
            BabyBear::from_u32(2154197895),
            BabyBear::from_u32(2459687029),
            BabyBear::from_u32(2870634489),
            BabyBear::from_u32(1375012945),
            BabyBear::from_u32(1529454825),
            BabyBear::from_u32(306140690),
            BabyBear::from_u32(2855578299),
            BabyBear::from_u32(1246997295),
            BabyBear::from_u32(3024298763),
            BabyBear::from_u32(1915270363),
            BabyBear::from_u32(1218245412),
            BabyBear::from_u32(2479314020),
            BabyBear::from_u32(2989827755),
            BabyBear::from_u32(814378556),
            BabyBear::from_u32(4039775921),
        ],
        [
            BabyBear::from_u32(1165280628),
            BabyBear::from_u32(1203983801),
            BabyBear::from_u32(3814740033),
            BabyBear::from_u32(1919627044),
            BabyBear::from_u32(600240215),
            BabyBear::from_u32(773269071),
            BabyBear::from_u32(486685186),
            BabyBear::from_u32(4254048810),
            BabyBear::from_u32(1415023565),
            BabyBear::from_u32(502840102),
            BabyBear::from_u32(4225648358),
            BabyBear::from_u32(510217063),
            BabyBear::from_u32(166444818),
            BabyBear::from_u32(1430745893),
            BabyBear::from_u32(1376516190),
            BabyBear::from_u32(1775891321),
        ]
    ];


    pub static ref RC_16_30_U32: [[u32; 16]; 21] = [
        [
            (2110014213),
            (3964964605),
            (2190662774),
            (2732996483),
            (640767983),
            (3403899136),
            (1716033721),
            (1606702601),
            (3759873288),
            (1466015491),
            (1498308946),
            (2844375094),
            (3042463841),
            (1969905919),
            (4109944726),
            (3925048366),
        ],
        [
            (3706859504),
            (759122502),
            (3167665446),
            (1131812921),
            (1080754908),
            (4080114493),
            (893583089),
            (2019677373),
            (3128604556),
            (580640471),
            (3277620260),
            (842931656),
            (548879852),
            (3608554714),
            (3575647916),
            (81826002),
        ],
        [
            (4289086263),
            (1563933798),
            (1440025885),
            (184445025),
            (2598651360),
            (1396647410),
            (1575877922),
            (3303853401),
            (137125468),
            (765010148),
            (633675867),
            (2037803363),
            (2573389828),
            (1895729703),
            (541515871),
            (1783382863),
        ],
        [
            (2641856484),
            (3035743342),
            (3672796326),
            (245668751),
            (2025460432),
            (201609705),
            (286217151),
            (4093475563),
            (2519572182),
            (3080699870),
            (2762001832),
            (1244250808),
            (606038199),
            (3182740831),
            (73007766),
            (2572204153),
        ],
        [
            (1196780786),
            (3447394443),
            (747167305),
            (2968073607),
            (1053214930),
            (1074411832),
            (4016794508),
            (1570312929),
            (113576933),
            (4042581186),
            (3634515733),
            (1032701597),
            (2364839308),
            (3840286918),
            (888378655),
            (2520191583),
        ],
        [
            (36046858),
            (2927525953),
            (3912129105),
            (4004832531),
            (193772436),
            (1590247392),
            (4125818172),
            (2516251696),
            (4050945750),
            (269498914),
            (1973292656),
            (891403491),
            (1845429189),
            (2611996363),
            (2310542653),
            (4071195740),
        ],
        [
            (3505307391),
            (786445290),
            (3815313971),
            (1111591756),
            (4233279834),
            (2775453034),
            (1991257625),
            (2940505809),
            (2751316206),
            (1028870679),
            (1282466273),
            (1059053371),
            (834521354),
            (138721483),
            (3100410803),
            (3843128331),
        ],
        [
            (3878220780),
            (4058162439),
            (1478942487),
            (799012923),
            (496734827),
            (3521261236),
            (755421082),
            (1361409515),
            (392099473),
            (3178453393),
            (4068463721),
            (7935614),
            (4140885645),
            (2150748066),
            (1685210312),
            (3852983224),
        ],
        [
            (2896943075),
            (3087590927),
            (992175959),
            (970216228),
            (3473630090),
            (3899670400),
            (3603388822),
            (2633488197),
            (2479406964),
            (2420952999),
            (1852516800),
            (4253075697),
            (979699862),
            (1163403191),
            (1608599874),
            (3056104448),
        ],
        [
            (3779109343),
            (536205958),
            (4183458361),
            (1649720295),
            (1444912244),
            (3122230878),
            (384301396),
            (4228198516),
            (1662916865),
            (4082161114),
            (2121897314),
            (1706239958),
            (4166959388),
            (1626054781),
            (3005858978),
            (1431907253),
        ],
        [
            (1418914503),
            (1365856753),
            (3942715745),
            (1429155552),
            (3545642795),
            (3772474257),
            (1621094396),
            (2154399145),
            (826697382),
            (1700781391),
            (3539164324),
            (652815039),
            (442484755),
            (2055299391),
            (1064289978),
            (1152335780),
        ],
        [
            (3417648695),
            (186040114),
            (3475580573),
            (2113941250),
            (1779573826),
            (1573808590),
            (3235694804),
            (2922195281),
            (1119462702),
            (3688305521),
            (1849567013),
            (667446787),
            (753897224),
            (1896396780),
            (3143026334),
            (3829603876),
        ],
        [
            (859661334),
            (3898844357),
            (180258337),
            (2321867017),
            (3599002504),
            (2886782421),
            (3038299378),
            (1035366250),
            (2038912197),
            (2920174523),
            (1277696101),
            (2785700290),
            (3806504335),
            (3518858933),
            (654843672),
            (2127120275),
        ],
        [
            (1548195514),
            (2378056027),
            (390914568),
            (1472049779),
            (1552596765),
            (1905886441),
            (1611959354),
            (3653263304),
            (3423946386),
            (340857935),
            (2208879480),
            (139364268),
            (3447281773),
            (3777813707),
            (55640413),
            (4101901741),
        ],
        [
            (104929687),
            (1459980974),
            (1831234737),
            (457139004),
            (2581487628),
            (2112044563),
            (3567013861),
            (2792004347),
            (576325418),
            (41126132),
            (2713562324),
            (151213722),
            (2891185935),
            (546846420),
            (2939794919),
            (2543469905)
        ],
        [
            (2191909784),
            (3315138460),
            (530414574),
            (1242280418),
            (1211740715),
            (3993672165),
            (2505083323),
            (3845798801),
            (538768466),
            (2063567560),
            (3366148274),
            (1449831887),
            (2408012466),
            (294726285),
            (3943435493),
            (924016661),
        ],
        [
            (3633138367),
            (3222789372),
            (809116305),
            (30100013),
            (2655172876),
            (2564247117),
            (2478649732),
            (4113689151),
            (4120146082),
            (2512308515),
            (650406041),
            (4240012393),
            (2683508708),
            (951073977),
            (3460081988),
            (339124269),
        ],
        [
            (130182653),
            (2755946749),
            (542600513),
            (2816103022),
            (1931786340),
            (2044470840),
            (1709908013),
            (2938369043),
            (3640399693),
            (1374470239),
            (2191149676),
            (2637495682),
            (4236394040),
            (2289358846),
            (3833368530),
            (974546524),
        ],
        [
            (3306659113),
            (2234814261),
            (1188782305),
            (223782844),
            (2248980567),
            (2309786141),
            (2023401627),
            (3278877413),
            (2022138149),
            (575851471),
            (1612560780),
            (3926656936),
            (3318548977),
            (2591863678),
            (188109355),
            (4217723909),
        ],
        [
            (1564209905),
            (2154197895),
            (2459687029),
            (2870634489),
            (1375012945),
            (1529454825),
            (306140690),
            (2855578299),
            (1246997295),
            (3024298763),
            (1915270363),
            (1218245412),
            (2479314020),
            (2989827755),
            (814378556),
            (4039775921),
        ],
        [
            (1165280628),
            (1203983801),
            (3814740033),
            (1919627044),
            (600240215),
            (773269071),
            (486685186),
            (4254048810),
            (1415023565),
            (502840102),
            (4225648358),
            (510217063),
            (166444818),
            (1430745893),
            (1376516190),
            (1775891321),
        ]
    ];
}

pub fn poseidon2_init(
) -> Poseidon2BabyBear<16> {
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 13;
    let mut round_constants = RC_16_30.to_vec();
    let internal_start = ROUNDS_F / 2;            //=4
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P; //=17
    let internal_round_constants =
        round_constants.drain(internal_start..internal_end).map(|vec| vec[0]).collect::<Vec<_>>();
    //let external_round_constants = round_constants;
    let external_round_constants = ExternalLayerConstants::new(
            round_constants[..(ROUNDS_F / 2)].to_vec(),
            round_constants[(ROUNDS_F / 2)..].to_vec(),
        );
    Poseidon2BabyBear::new(external_round_constants, internal_round_constants)
}

use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};

pub fn poseidon2_hash(input: Vec<BabyBear>) -> [BabyBear; 8] {
    POSEIDON2_HASHER.hash_iter(input)
}

pub fn poseidon2_hasher() -> PaddingFreeSponge<
    Poseidon2BabyBear<16>,
    16,
    8,
    8,
> {
    let hasher = poseidon2_init();
    PaddingFreeSponge::<
        Poseidon2BabyBear<16>,
        16,
        8,
        8,
    >::new(hasher)
}

lazy_static! {
    pub static ref POSEIDON2_HASHER: PaddingFreeSponge::<
        Poseidon2BabyBear<16>,
        16,
        8,
        8,
    > = poseidon2_hasher();
}

/// Append a single deferred proof to a hash chain of deferred proofs.
pub fn hash_deferred_proof(
    prev_digest: &[BabyBear; 8],
    vk_digest: &[BabyBear; 8],
    pv_digest: &[BabyBear; 32],
) -> [BabyBear; 8] {
    let mut inputs = Vec::with_capacity(48);
    inputs.extend_from_slice(prev_digest);
    inputs.extend_from_slice(vk_digest);
    inputs.extend_from_slice(pv_digest);
    poseidon2_hash(inputs.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    #[test]
    fn test_poseidon2() {
        let inputs = BabyBear::new_array([694761938, 1057063428, 79601855, 539810573, 472422960, 413992607, 117369146, 1370432434, 
                        0, 21185043, 90322736, 156256384, 23627556, 34171264, 126183783, 25645942, 
                        885797405, 1130275556, 567836311, 52700240, 239639200, 442612155, 1839439733, 19, 
                        524288, 1, 1049899240, 19, 524288, 1, 1049899240, 19, 
                        524288, 1, 1049899240, 18, 262144, 1, 1559589183, 17, 
                        131072, 1, 1286330022, 17, 131072, 1, 1286330022, 16, 
                        65536, 1, 1421947380, 16, 65536, 1, 1421947380, 4, 16, 1, 196396260]);
        let hash_ret = poseidon2_hash(inputs.to_vec());

        let expected: [BabyBear; 8] = BabyBear::new_array([344840138, 1920004139, 1145843621, 1099538827, 613084861, 1564249377, 178378012, 791822241]);
         
        assert_eq!(hash_ret, expected); 
    }

    use p3_symmetric::Permutation;
    #[test]
    fn test_permute() {
        let inputs = BabyBear::new_array([1984058442, 1779686813, 786767462, 334328488, 664932607, 1211726978, 653708563, 1908711429, 
                                        748182753, 1702519043, 182110445, 760024660, 807892063, 531542087, 1190845413, 1009472915]);
        let perm = poseidon2_init();
        let output = perm.permute(inputs);

        let expected = BabyBear::new_array([347216488, 1080055031, 427057322, 1709109579, 163565340, 1772928872, 652116498, 1067633747, 1731532717, 1424536438, 18136182, 1535395943, 1657769785, 1786735392, 1322137382, 35236760]);
         
        assert_eq!(output, expected); 
    }

    use p3_commit::Mmcs;
    use p3_commit::Pcs;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::TruncatedPermutation;
    use p3_matrix::Matrix;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;

    use p3_fri::{FriConfig,   TwoAdicFriPcs};
    use p3_merkle_tree::MerkleTreeMmcs;
    use rand_xoshiro::Xoroshiro128Plus;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    // Redefine types for testing
    type F = BabyBear;
    //type Perm = Poseidon2BabyBear<16>;
    type H = MyHash;
    type C = MyCompress;
    type P = <F as Field>::Packing;
    type PW = <F as Field>::Packing;

    #[test]
    fn test_merkle_commit() {
        let perm = poseidon2_init();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        
        // Create CPU and GPU versions of the MMCS
        let cpu_mmcs = p3_merkle_tree::MerkleTreeMmcs::<P, PW, H, C, DIGEST_SIZE>::new(hash.clone(), compress.clone());


        // Create some test matrices
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng, 512, 32);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 65536, 18);
        let mat3 = RowMajorMatrix::<F>::rand(&mut rng, 16384, 39);
        let mat4 = RowMajorMatrix::<F>::rand(&mut rng, 8192, 412);
       
        let start = std::time::Instant::now();
        let (cpu_root, _) = cpu_mmcs.commit(vec![mat1.clone(), mat2.clone(), mat3.clone(), mat4.clone()]);
         let duration = start.elapsed();
        println!("-- cpu commit , duration:{:?}", duration);
       
    }

    #[test]
    fn test_pcs_commit() {
        let perm = poseidon2_init();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let dft = Dft::default();
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let fri_config = FriConfig { 
                            log_blowup: 1, 
                            log_final_poly_len:0, 
                            num_queries: 28,
                            proof_of_work_bits: 16, 
                            mmcs: challenge_mmcs };
                            
        let pcs = MyPcs::new(dft, val_mmcs, fri_config);
        let mut domain_trace = Vec::new();

        //repeat 1
         let input_vec = BabyBear::new_array([2, 3, 4, 
                                                5, 6, 7, 
                                                334873, 334874, 334875, 
                                                334876, 334877, 334878, 
                                                334879, 334880, 334881, 
                                                0, 334882, 0, 
                                                334883, 0, 334884, 
                                                0, 334885, 0, 
                                                334886, 0, 334887, 
                                                0, 334888, 0, 
                                                334889, 1, 334890, 
                                                1, 334891, 1, 
                                                334892, 1, 334893, 
                                                1, 334894, 1, 
                                                334895, 1, 334896, 
                                                1, 2013265920, 9]);

        let trace = RowMajorMatrix::new(input_vec.to_vec(), 3);
      
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, trace.height());
        domain_trace.push((domain, trace));

        //r2
        let input_vec = BabyBear::new_array([334873, 334874, 334875, 334876, 
                                                334877, 334878, 334879, 334880, 
                                                334881, 0, 334882, 0, 
                                                334883, 0, 334884, 0, 
                                                334885, 0, 334886, 0, 
                                                334887, 0, 334888, 0, 
                                                334889, 1, 334890, 1, 
                                                334891, 1, 334892, 1, 
                                                334893, 1, 334894, 1, 
                                                334895, 1, 334896, 1, 
                                                8, 9, 10, 11, 
                                                12, 13, 14, 15, 
                                                334889, 334890, 334891, 334892, 
                                                334893, 334894, 334895, 334896, 
                                                334897, 0, 334898, 0, 
                                                334899, 0, 334900, 0]);

        let trace = RowMajorMatrix::new(input_vec.to_vec(), 4);
      
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, trace.height());
        domain_trace.push((domain, trace));

        //r3
        let input_vec = BabyBear::new_array([334877, 334878, 334879, 334880, 334881, 
                                            0, 334882, 0, 334883, 0, 
                                            334884, 0, 334885, 0, 334886, 
                                            0, 334887, 0, 334888, 0, 
                                            334889, 1, 334890, 1, 334891, 
                                            1, 334892, 1, 334893, 1, 
                                            334894, 1, 334895, 1, 334896, 
                                            8, 9, 10, 11, 12]);

        let trace = RowMajorMatrix::new(input_vec.to_vec(), 5);
      
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, trace.height());
        domain_trace.push((domain, trace));

        //r4
        let input_vec = BabyBear::new_array([334881, 0, 334882, 0, 334883, 0, 
                                            334884, 0, 334885, 0, 334886, 0, 
                                            334887, 0, 334888, 0, 334889, 1, 
                                            334890, 1, 334891, 1, 334892, 1]);

        let trace = RowMajorMatrix::new(input_vec.to_vec(), 6);
      
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, trace.height());
        domain_trace.push((domain, trace));

        //r5
        let input_vec = BabyBear::new_array([334884, 0, 334885, 0, 334886, 0, 334887, 0, 334888, 0, 
            334889, 1, 334890, 1, 334891, 1, 334892, 1, 334893, 1, 
            334894, 1, 334895, 1, 334896, 1, 2013265920,8, 9, 10, 
            11, 12, 13, 14, 15, 334889, 334890, 334891, 334892, 334893, 
            334894, 334895, 334896, 334897, 0, 334898, 0, 334899, 0, 334900, 
            0, 334901, 0, 334902, 0, 334903, 0, 334904, 0, 334905, 
            1, 334906, 1, 334907, 1, 334908, 1, 334909, 1, 334910, 
            1, 334911, 1, 334912, 1, 2013265920,16, 17, 18, 19]);

        let trace = RowMajorMatrix::new(input_vec.to_vec(), 10);
      
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, trace.height());
        domain_trace.push((domain, trace));
       
       println!("domain_trace.len={}", domain_trace.len());
        let (commit, data) = <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, domain_trace);

        let expected : [BabyBear; 8] = BabyBear::new_array([413309842, 1757403773, 426488468, 212647048, 1070719903, 1223953761, 63749309, 21512714]);
        
        assert_eq!(commit, expected);
    }

    use sha2::{Sha256, Digest};
    use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::File,};

    //type F = BabyBear;
    
    #[test]
    fn test_verify_vk_map() -> anyhow::Result<()> {
        let file = File::open("test_vk_map.bin")?;
        let loaded: BTreeMap<[F; 4], usize> = bincode::deserialize_from(file)?;
        println!("Loaded entries: {}", loaded.len());
        println!("Loaded keys: {:?}", loaded.keys());
        Ok(())
    }

    #[test]
    fn test_gen_vk_map_bin() -> anyhow::Result<()> {

        let vk_set: Vec<[F; 4]> = vec![
            [F::from_u32(1522375035), F::ONE, F::ONE, F::ZERO],
            [F::from_u32(1522375035), F::TWO, F::ONE, F::ZERO],
            [F::from_u32(1522375035), F::TWO, F::ONE, F::ONE],
            [F::from_u32(1522375035), F::TWO, F::ZERO, F::ONE],
        ];
        
  
        let vk_map = vk_set.into_iter()
            .enumerate()
            .map(|(i, vk)| (vk, i))
            .collect::<BTreeMap<_, _>>();

      
        let mut file = File::create("test_vk_map.bin")?; 
        bincode::serialize_into(&mut file, &vk_map)?;    
        
   
        let bytes = std::fs::read("test_vk_map.bin")?;
        println!("Generated file hash: {:x}", Sha256::digest(&bytes));
        
        Ok(())
    }
    
    pub const DIGEST_SIZE: usize = 4;
    use p3_field::PrimeField32;
    #[test]
    fn test_transfer_vk() -> anyhow::Result<()> {
        let file = File::open("test_vk_map.bin")?;  //generating by  org-sp1, 
        let allowed_vk_map: BTreeMap<[u32; 4], usize> = bincode::deserialize_from(file)?;
        println!("org allowed_vk_map entries: {}", allowed_vk_map.len());
        println!("org allowed_vk_map keys: {:?}", allowed_vk_map.keys());

        //
        let new_keys: Vec<[BabyBear; DIGEST_SIZE]> = allowed_vk_map
        .keys()
        .map(|original_key| {
            //let u32_array: [F; DIGEST_SIZE] = original_key.map(|x| F::from_u32(x));
           //u32_array
           original_key.map(|x| F::from_u32(x))
        })
        .collect();

        println!("new allowed_vk_map entries: {}", new_keys.len());
        println!("new allowed_vk_map keys: {:?}", new_keys);

        Ok(())
    }

}
