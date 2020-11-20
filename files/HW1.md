```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson
from natsort import natsorted
```


```python
df=pd.read_csv("merged.csv",error_bad_lines=False)

```

    b'Skipping line 382: expected 62 fields, saw 105\nSkipping line 383: expected 62 fields, saw 105\nSkipping line 384: expected 62 fields, saw 105\nSkipping line 385: expected 62 fields, saw 105\nSkipping line 386: expected 62 fields, saw 105\nSkipping line 387: expected 62 fields, saw 105\nSkipping line 388: expected 62 fields, saw 105\nSkipping line 389: expected 62 fields, saw 105\nSkipping line 390: expected 62 fields, saw 105\nSkipping line 391: expected 62 fields, saw 105\nSkipping line 392: expected 62 fields, saw 105\nSkipping line 393: expected 62 fields, saw 105\nSkipping line 394: expected 62 fields, saw 105\nSkipping line 395: expected 62 fields, saw 105\nSkipping line 396: expected 62 fields, saw 105\nSkipping line 397: expected 62 fields, saw 105\nSkipping line 398: expected 62 fields, saw 105\nSkipping line 399: expected 62 fields, saw 105\nSkipping line 400: expected 62 fields, saw 105\nSkipping line 401: expected 62 fields, saw 105\nSkipping line 402: expected 62 fields, saw 105\nSkipping line 403: expected 62 fields, saw 105\nSkipping line 404: expected 62 fields, saw 105\nSkipping line 405: expected 62 fields, saw 105\nSkipping line 406: expected 62 fields, saw 105\nSkipping line 407: expected 62 fields, saw 105\nSkipping line 408: expected 62 fields, saw 105\nSkipping line 409: expected 62 fields, saw 105\nSkipping line 410: expected 62 fields, saw 105\nSkipping line 411: expected 62 fields, saw 105\nSkipping line 412: expected 62 fields, saw 105\nSkipping line 413: expected 62 fields, saw 105\nSkipping line 414: expected 62 fields, saw 105\nSkipping line 415: expected 62 fields, saw 105\nSkipping line 416: expected 62 fields, saw 105\nSkipping line 417: expected 62 fields, saw 105\nSkipping line 418: expected 62 fields, saw 105\nSkipping line 419: expected 62 fields, saw 105\nSkipping line 420: expected 62 fields, saw 105\nSkipping line 421: expected 62 fields, saw 105\nSkipping line 422: expected 62 fields, saw 105\nSkipping line 423: expected 62 fields, saw 105\nSkipping line 424: expected 62 fields, saw 105\nSkipping line 425: expected 62 fields, saw 105\nSkipping line 426: expected 62 fields, saw 105\nSkipping line 427: expected 62 fields, saw 105\nSkipping line 428: expected 62 fields, saw 105\nSkipping line 429: expected 62 fields, saw 105\nSkipping line 430: expected 62 fields, saw 105\nSkipping line 431: expected 62 fields, saw 105\nSkipping line 432: expected 62 fields, saw 105\nSkipping line 433: expected 62 fields, saw 105\nSkipping line 434: expected 62 fields, saw 105\nSkipping line 435: expected 62 fields, saw 105\nSkipping line 436: expected 62 fields, saw 105\nSkipping line 437: expected 62 fields, saw 105\nSkipping line 438: expected 62 fields, saw 105\nSkipping line 439: expected 62 fields, saw 105\nSkipping line 440: expected 62 fields, saw 105\nSkipping line 441: expected 62 fields, saw 105\nSkipping line 442: expected 62 fields, saw 105\nSkipping line 443: expected 62 fields, saw 105\nSkipping line 444: expected 62 fields, saw 105\nSkipping line 445: expected 62 fields, saw 105\nSkipping line 446: expected 62 fields, saw 105\nSkipping line 447: expected 62 fields, saw 105\nSkipping line 448: expected 62 fields, saw 105\nSkipping line 449: expected 62 fields, saw 105\nSkipping line 450: expected 62 fields, saw 105\nSkipping line 451: expected 62 fields, saw 105\nSkipping line 452: expected 62 fields, saw 105\nSkipping line 453: expected 62 fields, saw 105\nSkipping line 454: expected 62 fields, saw 105\nSkipping line 455: expected 62 fields, saw 105\nSkipping line 456: expected 62 fields, saw 105\nSkipping line 457: expected 62 fields, saw 105\nSkipping line 458: expected 62 fields, saw 105\nSkipping line 459: expected 62 fields, saw 105\nSkipping line 460: expected 62 fields, saw 105\nSkipping line 461: expected 62 fields, saw 105\nSkipping line 462: expected 62 fields, saw 105\nSkipping line 463: expected 62 fields, saw 105\nSkipping line 464: expected 62 fields, saw 105\nSkipping line 465: expected 62 fields, saw 105\nSkipping line 466: expected 62 fields, saw 105\nSkipping line 467: expected 62 fields, saw 105\nSkipping line 468: expected 62 fields, saw 105\nSkipping line 469: expected 62 fields, saw 105\nSkipping line 470: expected 62 fields, saw 105\nSkipping line 471: expected 62 fields, saw 105\nSkipping line 472: expected 62 fields, saw 105\nSkipping line 473: expected 62 fields, saw 105\nSkipping line 474: expected 62 fields, saw 105\nSkipping line 475: expected 62 fields, saw 105\nSkipping line 476: expected 62 fields, saw 105\nSkipping line 477: expected 62 fields, saw 105\nSkipping line 478: expected 62 fields, saw 105\nSkipping line 479: expected 62 fields, saw 105\nSkipping line 480: expected 62 fields, saw 105\nSkipping line 481: expected 62 fields, saw 105\nSkipping line 482: expected 62 fields, saw 105\nSkipping line 483: expected 62 fields, saw 105\nSkipping line 484: expected 62 fields, saw 105\nSkipping line 485: expected 62 fields, saw 105\nSkipping line 486: expected 62 fields, saw 105\nSkipping line 487: expected 62 fields, saw 105\nSkipping line 488: expected 62 fields, saw 105\nSkipping line 489: expected 62 fields, saw 105\nSkipping line 490: expected 62 fields, saw 105\nSkipping line 491: expected 62 fields, saw 105\nSkipping line 492: expected 62 fields, saw 105\nSkipping line 493: expected 62 fields, saw 105\nSkipping line 494: expected 62 fields, saw 105\nSkipping line 495: expected 62 fields, saw 105\nSkipping line 496: expected 62 fields, saw 105\nSkipping line 497: expected 62 fields, saw 105\nSkipping line 498: expected 62 fields, saw 105\nSkipping line 499: expected 62 fields, saw 105\nSkipping line 500: expected 62 fields, saw 105\nSkipping line 501: expected 62 fields, saw 105\nSkipping line 502: expected 62 fields, saw 105\nSkipping line 503: expected 62 fields, saw 105\nSkipping line 504: expected 62 fields, saw 105\nSkipping line 505: expected 62 fields, saw 105\nSkipping line 506: expected 62 fields, saw 105\nSkipping line 507: expected 62 fields, saw 105\nSkipping line 508: expected 62 fields, saw 105\nSkipping line 509: expected 62 fields, saw 105\nSkipping line 510: expected 62 fields, saw 105\nSkipping line 511: expected 62 fields, saw 105\nSkipping line 512: expected 62 fields, saw 105\nSkipping line 513: expected 62 fields, saw 105\nSkipping line 514: expected 62 fields, saw 105\nSkipping line 515: expected 62 fields, saw 105\nSkipping line 516: expected 62 fields, saw 105\nSkipping line 517: expected 62 fields, saw 105\nSkipping line 518: expected 62 fields, saw 105\nSkipping line 519: expected 62 fields, saw 105\nSkipping line 520: expected 62 fields, saw 105\nSkipping line 521: expected 62 fields, saw 105\nSkipping line 522: expected 62 fields, saw 105\nSkipping line 523: expected 62 fields, saw 105\nSkipping line 524: expected 62 fields, saw 105\nSkipping line 525: expected 62 fields, saw 105\nSkipping line 526: expected 62 fields, saw 105\nSkipping line 527: expected 62 fields, saw 105\nSkipping line 528: expected 62 fields, saw 105\nSkipping line 529: expected 62 fields, saw 105\nSkipping line 530: expected 62 fields, saw 105\nSkipping line 531: expected 62 fields, saw 105\nSkipping line 532: expected 62 fields, saw 105\nSkipping line 533: expected 62 fields, saw 105\nSkipping line 534: expected 62 fields, saw 105\nSkipping line 535: expected 62 fields, saw 105\nSkipping line 536: expected 62 fields, saw 105\nSkipping line 537: expected 62 fields, saw 105\nSkipping line 538: expected 62 fields, saw 105\nSkipping line 539: expected 62 fields, saw 105\nSkipping line 540: expected 62 fields, saw 105\nSkipping line 541: expected 62 fields, saw 105\nSkipping line 542: expected 62 fields, saw 105\nSkipping line 543: expected 62 fields, saw 105\nSkipping line 544: expected 62 fields, saw 105\nSkipping line 545: expected 62 fields, saw 105\nSkipping line 546: expected 62 fields, saw 105\nSkipping line 547: expected 62 fields, saw 105\nSkipping line 548: expected 62 fields, saw 105\nSkipping line 549: expected 62 fields, saw 105\nSkipping line 550: expected 62 fields, saw 105\nSkipping line 551: expected 62 fields, saw 105\nSkipping line 552: expected 62 fields, saw 105\nSkipping line 553: expected 62 fields, saw 105\nSkipping line 554: expected 62 fields, saw 105\nSkipping line 555: expected 62 fields, saw 105\nSkipping line 556: expected 62 fields, saw 105\nSkipping line 557: expected 62 fields, saw 105\nSkipping line 558: expected 62 fields, saw 105\nSkipping line 559: expected 62 fields, saw 105\nSkipping line 560: expected 62 fields, saw 105\nSkipping line 561: expected 62 fields, saw 105\nSkipping line 562: expected 62 fields, saw 105\nSkipping line 563: expected 62 fields, saw 105\nSkipping line 564: expected 62 fields, saw 105\nSkipping line 565: expected 62 fields, saw 105\nSkipping line 566: expected 62 fields, saw 105\nSkipping line 567: expected 62 fields, saw 105\nSkipping line 568: expected 62 fields, saw 105\nSkipping line 569: expected 62 fields, saw 105\nSkipping line 570: expected 62 fields, saw 105\nSkipping line 571: expected 62 fields, saw 105\nSkipping line 572: expected 62 fields, saw 105\nSkipping line 573: expected 62 fields, saw 105\nSkipping line 574: expected 62 fields, saw 105\nSkipping line 575: expected 62 fields, saw 105\nSkipping line 576: expected 62 fields, saw 105\nSkipping line 577: expected 62 fields, saw 105\nSkipping line 578: expected 62 fields, saw 105\nSkipping line 579: expected 62 fields, saw 105\nSkipping line 580: expected 62 fields, saw 105\nSkipping line 581: expected 62 fields, saw 105\nSkipping line 582: expected 62 fields, saw 105\nSkipping line 583: expected 62 fields, saw 105\nSkipping line 584: expected 62 fields, saw 105\nSkipping line 585: expected 62 fields, saw 105\nSkipping line 586: expected 62 fields, saw 105\nSkipping line 587: expected 62 fields, saw 105\nSkipping line 588: expected 62 fields, saw 105\nSkipping line 589: expected 62 fields, saw 105\nSkipping line 590: expected 62 fields, saw 105\nSkipping line 591: expected 62 fields, saw 105\nSkipping line 592: expected 62 fields, saw 105\nSkipping line 593: expected 62 fields, saw 105\nSkipping line 594: expected 62 fields, saw 105\nSkipping line 595: expected 62 fields, saw 105\nSkipping line 596: expected 62 fields, saw 105\nSkipping line 597: expected 62 fields, saw 105\nSkipping line 598: expected 62 fields, saw 105\nSkipping line 599: expected 62 fields, saw 105\nSkipping line 600: expected 62 fields, saw 105\nSkipping line 601: expected 62 fields, saw 105\nSkipping line 602: expected 62 fields, saw 105\nSkipping line 603: expected 62 fields, saw 105\nSkipping line 604: expected 62 fields, saw 105\nSkipping line 605: expected 62 fields, saw 105\nSkipping line 606: expected 62 fields, saw 105\nSkipping line 607: expected 62 fields, saw 105\nSkipping line 608: expected 62 fields, saw 105\nSkipping line 609: expected 62 fields, saw 105\nSkipping line 610: expected 62 fields, saw 105\nSkipping line 611: expected 62 fields, saw 105\nSkipping line 612: expected 62 fields, saw 105\nSkipping line 613: expected 62 fields, saw 105\nSkipping line 614: expected 62 fields, saw 105\nSkipping line 615: expected 62 fields, saw 105\nSkipping line 616: expected 62 fields, saw 105\nSkipping line 617: expected 62 fields, saw 105\nSkipping line 618: expected 62 fields, saw 105\nSkipping line 619: expected 62 fields, saw 105\nSkipping line 620: expected 62 fields, saw 105\nSkipping line 621: expected 62 fields, saw 105\nSkipping line 622: expected 62 fields, saw 105\nSkipping line 623: expected 62 fields, saw 105\nSkipping line 624: expected 62 fields, saw 105\nSkipping line 625: expected 62 fields, saw 105\nSkipping line 626: expected 62 fields, saw 105\nSkipping line 627: expected 62 fields, saw 105\nSkipping line 628: expected 62 fields, saw 105\nSkipping line 629: expected 62 fields, saw 105\nSkipping line 630: expected 62 fields, saw 105\nSkipping line 631: expected 62 fields, saw 105\nSkipping line 632: expected 62 fields, saw 105\nSkipping line 633: expected 62 fields, saw 105\nSkipping line 634: expected 62 fields, saw 105\nSkipping line 635: expected 62 fields, saw 105\nSkipping line 636: expected 62 fields, saw 105\nSkipping line 637: expected 62 fields, saw 105\nSkipping line 638: expected 62 fields, saw 105\nSkipping line 639: expected 62 fields, saw 105\nSkipping line 640: expected 62 fields, saw 105\nSkipping line 641: expected 62 fields, saw 105\nSkipping line 642: expected 62 fields, saw 105\nSkipping line 643: expected 62 fields, saw 105\nSkipping line 644: expected 62 fields, saw 105\nSkipping line 645: expected 62 fields, saw 105\nSkipping line 646: expected 62 fields, saw 105\nSkipping line 647: expected 62 fields, saw 105\nSkipping line 648: expected 62 fields, saw 105\nSkipping line 649: expected 62 fields, saw 105\nSkipping line 650: expected 62 fields, saw 105\nSkipping line 651: expected 62 fields, saw 105\nSkipping line 652: expected 62 fields, saw 105\nSkipping line 653: expected 62 fields, saw 105\nSkipping line 654: expected 62 fields, saw 105\nSkipping line 655: expected 62 fields, saw 105\nSkipping line 656: expected 62 fields, saw 105\nSkipping line 657: expected 62 fields, saw 105\nSkipping line 658: expected 62 fields, saw 105\nSkipping line 659: expected 62 fields, saw 105\nSkipping line 660: expected 62 fields, saw 105\nSkipping line 661: expected 62 fields, saw 105\nSkipping line 662: expected 62 fields, saw 105\nSkipping line 663: expected 62 fields, saw 105\nSkipping line 664: expected 62 fields, saw 105\nSkipping line 665: expected 62 fields, saw 105\nSkipping line 666: expected 62 fields, saw 105\nSkipping line 667: expected 62 fields, saw 105\nSkipping line 668: expected 62 fields, saw 105\nSkipping line 669: expected 62 fields, saw 105\nSkipping line 670: expected 62 fields, saw 105\nSkipping line 671: expected 62 fields, saw 105\nSkipping line 672: expected 62 fields, saw 105\nSkipping line 673: expected 62 fields, saw 105\nSkipping line 674: expected 62 fields, saw 105\nSkipping line 675: expected 62 fields, saw 105\nSkipping line 676: expected 62 fields, saw 105\nSkipping line 677: expected 62 fields, saw 105\nSkipping line 678: expected 62 fields, saw 105\nSkipping line 679: expected 62 fields, saw 105\nSkipping line 680: expected 62 fields, saw 105\nSkipping line 681: expected 62 fields, saw 105\nSkipping line 682: expected 62 fields, saw 105\nSkipping line 683: expected 62 fields, saw 105\nSkipping line 684: expected 62 fields, saw 105\nSkipping line 685: expected 62 fields, saw 105\nSkipping line 686: expected 62 fields, saw 105\nSkipping line 687: expected 62 fields, saw 105\nSkipping line 688: expected 62 fields, saw 105\nSkipping line 689: expected 62 fields, saw 105\nSkipping line 690: expected 62 fields, saw 105\nSkipping line 691: expected 62 fields, saw 105\nSkipping line 692: expected 62 fields, saw 105\nSkipping line 693: expected 62 fields, saw 105\nSkipping line 694: expected 62 fields, saw 105\nSkipping line 695: expected 62 fields, saw 105\nSkipping line 696: expected 62 fields, saw 105\nSkipping line 697: expected 62 fields, saw 105\nSkipping line 698: expected 62 fields, saw 105\nSkipping line 699: expected 62 fields, saw 105\nSkipping line 700: expected 62 fields, saw 105\nSkipping line 701: expected 62 fields, saw 105\nSkipping line 702: expected 62 fields, saw 105\nSkipping line 703: expected 62 fields, saw 105\nSkipping line 704: expected 62 fields, saw 105\nSkipping line 705: expected 62 fields, saw 105\nSkipping line 706: expected 62 fields, saw 105\nSkipping line 707: expected 62 fields, saw 105\nSkipping line 708: expected 62 fields, saw 105\nSkipping line 709: expected 62 fields, saw 105\nSkipping line 710: expected 62 fields, saw 105\nSkipping line 711: expected 62 fields, saw 105\nSkipping line 712: expected 62 fields, saw 105\nSkipping line 713: expected 62 fields, saw 105\nSkipping line 714: expected 62 fields, saw 105\nSkipping line 715: expected 62 fields, saw 105\nSkipping line 716: expected 62 fields, saw 105\nSkipping line 717: expected 62 fields, saw 105\nSkipping line 718: expected 62 fields, saw 105\nSkipping line 719: expected 62 fields, saw 105\nSkipping line 720: expected 62 fields, saw 105\nSkipping line 721: expected 62 fields, saw 105\nSkipping line 722: expected 62 fields, saw 105\nSkipping line 723: expected 62 fields, saw 105\nSkipping line 724: expected 62 fields, saw 105\nSkipping line 725: expected 62 fields, saw 105\nSkipping line 726: expected 62 fields, saw 105\nSkipping line 727: expected 62 fields, saw 105\nSkipping line 728: expected 62 fields, saw 105\nSkipping line 729: expected 62 fields, saw 105\nSkipping line 730: expected 62 fields, saw 105\nSkipping line 731: expected 62 fields, saw 105\nSkipping line 732: expected 62 fields, saw 105\nSkipping line 733: expected 62 fields, saw 105\nSkipping line 734: expected 62 fields, saw 105\nSkipping line 735: expected 62 fields, saw 105\nSkipping line 736: expected 62 fields, saw 105\nSkipping line 737: expected 62 fields, saw 105\nSkipping line 738: expected 62 fields, saw 105\nSkipping line 739: expected 62 fields, saw 105\nSkipping line 740: expected 62 fields, saw 105\nSkipping line 741: expected 62 fields, saw 105\nSkipping line 742: expected 62 fields, saw 105\nSkipping line 743: expected 62 fields, saw 105\nSkipping line 744: expected 62 fields, saw 105\nSkipping line 745: expected 62 fields, saw 105\nSkipping line 746: expected 62 fields, saw 105\nSkipping line 747: expected 62 fields, saw 105\nSkipping line 748: expected 62 fields, saw 105\nSkipping line 749: expected 62 fields, saw 105\nSkipping line 750: expected 62 fields, saw 105\nSkipping line 751: expected 62 fields, saw 105\nSkipping line 752: expected 62 fields, saw 105\nSkipping line 753: expected 62 fields, saw 105\nSkipping line 754: expected 62 fields, saw 105\nSkipping line 755: expected 62 fields, saw 105\nSkipping line 756: expected 62 fields, saw 105\nSkipping line 757: expected 62 fields, saw 105\nSkipping line 758: expected 62 fields, saw 105\nSkipping line 759: expected 62 fields, saw 105\nSkipping line 760: expected 62 fields, saw 105\nSkipping line 761: expected 62 fields, saw 105\nSkipping line 762: expected 62 fields, saw 105\nSkipping line 763: expected 62 fields, saw 105\nSkipping line 764: expected 62 fields, saw 105\nSkipping line 765: expected 62 fields, saw 105\nSkipping line 766: expected 62 fields, saw 105\nSkipping line 767: expected 62 fields, saw 105\nSkipping line 768: expected 62 fields, saw 105\nSkipping line 769: expected 62 fields, saw 105\nSkipping line 770: expected 62 fields, saw 105\nSkipping line 771: expected 62 fields, saw 105\nSkipping line 772: expected 62 fields, saw 105\nSkipping line 773: expected 62 fields, saw 105\nSkipping line 774: expected 62 fields, saw 105\nSkipping line 775: expected 62 fields, saw 105\nSkipping line 776: expected 62 fields, saw 105\nSkipping line 777: expected 62 fields, saw 105\nSkipping line 778: expected 62 fields, saw 105\nSkipping line 779: expected 62 fields, saw 105\nSkipping line 780: expected 62 fields, saw 105\nSkipping line 781: expected 62 fields, saw 105\nSkipping line 782: expected 62 fields, saw 105\nSkipping line 783: expected 62 fields, saw 105\nSkipping line 784: expected 62 fields, saw 105\nSkipping line 785: expected 62 fields, saw 105\nSkipping line 786: expected 62 fields, saw 105\nSkipping line 787: expected 62 fields, saw 105\nSkipping line 788: expected 62 fields, saw 105\nSkipping line 789: expected 62 fields, saw 105\nSkipping line 790: expected 62 fields, saw 105\nSkipping line 791: expected 62 fields, saw 105\nSkipping line 792: expected 62 fields, saw 105\nSkipping line 793: expected 62 fields, saw 105\nSkipping line 794: expected 62 fields, saw 105\nSkipping line 795: expected 62 fields, saw 105\nSkipping line 796: expected 62 fields, saw 105\nSkipping line 797: expected 62 fields, saw 105\nSkipping line 798: expected 62 fields, saw 105\nSkipping line 799: expected 62 fields, saw 105\nSkipping line 800: expected 62 fields, saw 105\nSkipping line 801: expected 62 fields, saw 105\nSkipping line 802: expected 62 fields, saw 105\nSkipping line 803: expected 62 fields, saw 105\nSkipping line 804: expected 62 fields, saw 105\nSkipping line 805: expected 62 fields, saw 105\nSkipping line 806: expected 62 fields, saw 105\nSkipping line 807: expected 62 fields, saw 105\nSkipping line 808: expected 62 fields, saw 105\nSkipping line 809: expected 62 fields, saw 105\nSkipping line 810: expected 62 fields, saw 105\nSkipping line 811: expected 62 fields, saw 105\nSkipping line 812: expected 62 fields, saw 105\nSkipping line 813: expected 62 fields, saw 105\nSkipping line 814: expected 62 fields, saw 105\nSkipping line 815: expected 62 fields, saw 105\nSkipping line 816: expected 62 fields, saw 105\nSkipping line 817: expected 62 fields, saw 105\nSkipping line 818: expected 62 fields, saw 105\nSkipping line 819: expected 62 fields, saw 105\nSkipping line 820: expected 62 fields, saw 105\nSkipping line 821: expected 62 fields, saw 105\nSkipping line 822: expected 62 fields, saw 105\nSkipping line 823: expected 62 fields, saw 105\nSkipping line 824: expected 62 fields, saw 105\nSkipping line 825: expected 62 fields, saw 105\nSkipping line 826: expected 62 fields, saw 105\nSkipping line 827: expected 62 fields, saw 105\nSkipping line 828: expected 62 fields, saw 105\nSkipping line 829: expected 62 fields, saw 105\nSkipping line 830: expected 62 fields, saw 105\nSkipping line 831: expected 62 fields, saw 105\nSkipping line 832: expected 62 fields, saw 105\nSkipping line 833: expected 62 fields, saw 105\nSkipping line 834: expected 62 fields, saw 105\nSkipping line 835: expected 62 fields, saw 105\nSkipping line 836: expected 62 fields, saw 105\nSkipping line 837: expected 62 fields, saw 105\nSkipping line 838: expected 62 fields, saw 105\nSkipping line 839: expected 62 fields, saw 105\n'
    


```python
#Task 1
```


```python
df.shape
```




    (380, 62)




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Div</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>...</th>
      <th>BbAv&lt;2.5</th>
      <th>BbAH</th>
      <th>BbAHh</th>
      <th>BbMxAHH</th>
      <th>BbAvAHH</th>
      <th>BbMxAHA</th>
      <th>BbAvAHA</th>
      <th>PSCH</th>
      <th>PSCD</th>
      <th>PSCA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E0</td>
      <td>10/08/2018</td>
      <td>Man United</td>
      <td>Leicester</td>
      <td>2</td>
      <td>1</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>1.79</td>
      <td>17</td>
      <td>-0.75</td>
      <td>1.75</td>
      <td>1.70</td>
      <td>2.29</td>
      <td>2.21</td>
      <td>1.55</td>
      <td>4.07</td>
      <td>7.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Bournemouth</td>
      <td>Cardiff</td>
      <td>2</td>
      <td>0</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>1.83</td>
      <td>20</td>
      <td>-0.75</td>
      <td>2.20</td>
      <td>2.13</td>
      <td>1.80</td>
      <td>1.75</td>
      <td>1.88</td>
      <td>3.61</td>
      <td>4.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Fulham</td>
      <td>Crystal Palace</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>1.87</td>
      <td>22</td>
      <td>-0.25</td>
      <td>2.18</td>
      <td>2.11</td>
      <td>1.81</td>
      <td>1.77</td>
      <td>2.62</td>
      <td>3.38</td>
      <td>2.90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Huddersfield</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>3</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>1.84</td>
      <td>23</td>
      <td>1.00</td>
      <td>1.84</td>
      <td>1.80</td>
      <td>2.13</td>
      <td>2.06</td>
      <td>7.24</td>
      <td>3.95</td>
      <td>1.58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Newcastle</td>
      <td>Tottenham</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>1.81</td>
      <td>20</td>
      <td>0.25</td>
      <td>2.20</td>
      <td>2.12</td>
      <td>1.80</td>
      <td>1.76</td>
      <td>4.74</td>
      <td>3.53</td>
      <td>1.89</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 62 columns</p>
</div>




```python
k=sns.distplot(df.FTHG,kde=False,bins=7);
k.set_xlabel("Home Goals")
k.set_ylabel("Number of Games")
k.set_title("Home Score(Goals)");
```


![png](output_6_0.png)



```python
s=sns.distplot(df.FTAG,kde=False,bins=7);
s.set_xlabel("Away Goals")
s.set_ylabel("Number of Games")
s.set_title("Away Score (Goals)");
```


![png](output_7_0.png)



```python
m=sns.distplot(df.FTHG-df.FTAG,kde=False,bins=10);
m.set_xlabel("Home Goals-Away Goals")
m.set_ylabel("Number of Games")
m.set_title("Home Score(Goals)– Away Score(Goals)");
```


![png](output_8_0.png)



```python
df["FTHG"].value_counts()
```




    1    116
    2     95
    0     88
    3     48
    4     22
    5      8
    6      3
    Name: FTHG, dtype: int64




```python
df["FTAG"].mean()    
```




    1.2526315789473683




```python
df["FTHG"].mean()
```




    1.568421052631579




```python
x=df["FTAG"]
bins = 7
n, bins_edges,patches = plt.hist(x,bins,density=1, facecolor='darkblue', ec='white', log=0)
mu=df["FTAG"].mean()
x_plot = np.arange(0, max(df["FTAG"]+1))
plt.plot(x_plot, poisson.pmf(x_plot, mu), label='Poisson')
plt.title('Poisson Fit on FTAG')
plt.show()
```


![png](output_12_0.png)



```python
x2=df["FTHG"]
bins = 7
n, bins_edges,patches = plt.hist(x2,bins,density=1, facecolor='darkblue', ec='white', log=0)
mu2=df["FTHG"].mean()
x2_plot = np.arange(0, max(df["FTHG"]+1))
plt.plot(x2_plot, poisson.pmf(x2_plot, mu2), label='Poisson')
plt.title('Poisson Fit on FTHG')
plt.show()
```


![png](output_13_0.png)



```python
#Task 2
```


```python
#BW
```


```python
bwh=1/df["BWH"]
```


```python
bwh.head()
```




    0    0.653595
    1    0.526316
    2    0.408163
    3    0.160000
    4    0.263158
    Name: BWH, dtype: float64




```python
bwd=1/df["BWD"]
```


```python
bwd.head()
```




    0    0.250000
    1    0.294118
    2    0.303030
    3    0.256410
    4    0.285714
    Name: BWD, dtype: float64




```python
bwa=1/df["BWA"]
bwa.head()
```




    0    0.133333
    1    0.227273
    2    0.338983
    3    0.636943
    4    0.500000
    Name: BWA, dtype: float64




```python
#Calculating BW values using Normalization
```


```python
tbw=bwh+bwa+bwd
```


```python
tbw_=1/tbw
```


```python
new_bwh=tbw_*bwh
```


```python
new_bwd=tbw_*bwd
```


```python
new_bwa=tbw_*bwa
```


```python
#IW
```


```python
iwh=1/df["IWH"]
iwh.head()
```




    0    0.645161
    1    0.526316
    2    0.416667
    3    0.161290
    4    0.270270
    Name: IWH, dtype: float64




```python
iwd=1/df["IWD"]
iwd.head()
```




    0    0.263158
    1    0.285714
    2    0.303030
    3    0.250000
    4    0.298507
    Name: IWD, dtype: float64




```python
iwa=1/df["IWA"]
iwa.head()
```




    0    0.142857
    1    0.243902
    2    0.338983
    3    0.645161
    4    0.487805
    Name: IWA, dtype: float64




```python
#Calculating IW values using Normalization
```


```python
tiw=iwh+iwd+iwa
```


```python
tiw_=1/tiw
```


```python
new_iwh=tiw_*iwh
```


```python
new_iwd=tiw_*iwd
```


```python
new_iwa=tiw_*iwa
```


```python
#PS
```


```python
psh=1/df["PSH"]
psh.head()
```




    0    0.632911
    1    0.529101
    2    0.400000
    3    0.156006
    4    0.261097
    Name: PSH, dtype: float64




```python
psd=1/df["PSD"]
psd.head()
```




    0    0.254453
    1    0.275482
    2    0.289017
    3    0.248756
    4    0.280112
    Name: PSD, dtype: float64




```python
psa=1/df["PSA"]
psa.head()
```




    0    0.133333
    1    0.218341
    2    0.333333
    3    0.617284
    4    0.480769
    Name: PSA, dtype: float64




```python
#Calculating PS values using Normalization
```


```python
tps=psh+psd+psa
```


```python
tps_=1/tps
```


```python
new_psh=tps_*psh
```


```python
new_psd=tps_*psd
```


```python
new_psa=tps_*psa
```


```python
#WH
```


```python
whh=1/df["WHH"]
whh.head()
```




    0    0.636943
    1    0.523560
    2    0.408163
    3    0.172414
    4    0.263158
    Name: WHH, dtype: float64




```python
whd=1/df["WHD"]
whd.head()
```




    0    0.263158
    1    0.285714
    2    0.303030
    3    0.256410
    4    0.312500
    Name: WHD, dtype: float64




```python
wha=1/df["WHA"]
wha.head()
```




    0    0.166667
    1    0.250000
    2    0.357143
    3    0.636943
    4    0.487805
    Name: WHA, dtype: float64




```python
#Calculating WH values using Normalization
```


```python
twh=whh+wha+whd
```


```python
twh_=1/twh
```


```python
new_whh=twh_*whh
```


```python
new_whd=twh_*whd
```


```python
new_wha=twh_*wha
```


```python
#Task 2 Q3
```


```python
df["wh_prob"]=new_whh-new_wha
```


```python
df["wh_prob"]
```




    0      0.440842
    1      0.258252
    2      0.047757
    3     -0.435864
    4     -0.211241
             ...   
    375    0.652993
    376    0.664234
    377    0.561325
    378    0.181818
    379    0.125993
    Name: wh_prob, Length: 380, dtype: float64




```python
wh1=df[(df["FTR"]=="D") & (df["wh_prob"]>=-1)  & (df["wh_prob"]<-0.8)].index   #Takimng matches resulted with "Draw" for each corresponding bin

wh2=df[(df["FTR"]=="D") & (df["wh_prob"]>=-0.8)  & (df["wh_prob"]<-0.6)].index    

wh3=df[(df["FTR"]=="D") & (df["wh_prob"]>=-0.6)  & (df["wh_prob"]<-0.4)].index

wh4=df[(df["FTR"]=="D") & (df["wh_prob"]>=-0.4)  & (df["wh_prob"]<-0.2)].index

wh5=df[(df["FTR"]=="D") & (df["wh_prob"]>=-0.2)  & (df["wh_prob"]<-2.22e-16)].index

wh6=df[(df["FTR"]=="D") & (df["wh_prob"]>=-2.22e-16)  & (df["wh_prob"]<0.2)].index

wh7=df[(df["FTR"]=="D") & (df["wh_prob"]>=0.2)  & (df["wh_prob"]<0.4)].index

wh8=df[(df["FTR"]=="D") & (df["wh_prob"]>=0.4)  & (df["wh_prob"]<0.6)].index

wh9=df[(df["FTR"]=="D") & (df["wh_prob"]>=0.6)  & (df["wh_prob"]<0.8)].index

wh10=df[(df["FTR"]=="D") & (df["wh_prob"]>=0.8)  & (df["wh_prob"]<1)].index

```


```python
wh_rows = [[len(wh1)],[len(wh2)],[len(wh3)],[len(wh4)],[(len(wh5))],[len(wh6)],[len(wh7)],[len(wh8)],[len(wh9)],[len(wh10)]]
```


```python
df_wh = pd.DataFrame(wh_rows, columns=["num_wh"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
wh_rng,bins=pd.cut((new_whh-new_wha),bins=bin_range,include_lowest=True, right=False,retbins=True)  #making bins
```


```python
sorted_wh=wh_rng.value_counts(sort=True)                         #sorting bins
sorted_wh = sorted_wh.reindex(natsorted(sorted_wh.index))
```


```python
print(sorted_wh)   
```

    [-1.0, -0.8)          1
    [-0.8, -0.6)         20
    [-0.6, -0.4)         24
    [-0.4, -0.2)         37
    [-0.2, -2.22e-16)    48
    [-2.22e-16, 0.2)     84
    [0.2, 0.4)           58
    [0.4, 0.6)           53
    [0.6, 0.8)           44
    [0.8, 1.0)           11
    dtype: int64
    


```python
df_sorted_wh = sorted_wh.to_frame("counts")
```


```python
df_wh.reset_index(drop=True, inplace=True)
df_sorted_wh.reset_index(drop=True, inplace=True)
```


```python
actl_prob_wh=df_wh["num_wh"]/df_sorted_wh["counts"]   #Number of games resulted "Draw" dividing by total number of games corresponding bin
```


```python
actl_prob_wh
```




    0    0.000000
    1    0.100000
    2    0.083333
    3    0.135135
    4    0.187500
    5    0.285714
    6    0.241379
    7    0.207547
    8    0.090909
    9    0.000000
    dtype: float64




```python
pointswh = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]  #the middle of intervals was chosen as point.
```


```python
wh_points = pd.DataFrame(pointswh, columns=["points_wh"])
```


```python
a=sns.scatterplot(x=new_whh-new_wha,y=new_whd)
b=sns.scatterplot(x=wh_points["points_wh"],y=actl_prob_wh)
a.set_xlabel("P(Home Win)-P(Away Win)")
a.set_ylabel("P(Draw)")
a.set_title("WH");
```


![png](output_72_0.png)



```python

```


```python
df["ps_prob"]=new_psh-new_psa
```


```python
df["ps_prob"]
```




    0      0.489448
    1      0.303796
    2      0.065209
    3     -0.451328
    4     -0.214949
             ...   
    375    0.648039
    376    0.658818
    377    0.545368
    378    0.196429
    379    0.139420
    Name: ps_prob, Length: 380, dtype: float64




```python
ps1=df[(df["FTR"]=="D") & (df["ps_prob"]>=-1)  & (df["ps_prob"]<-0.8)].index

ps2=df[(df["FTR"]=="D") & (df["ps_prob"]>=-0.8)  & (df["ps_prob"]<-0.6)].index

ps3=df[(df["FTR"]=="D") & (df["ps_prob"]>=-0.6)  & (df["ps_prob"]<-0.4)].index

ps4=df[(df["FTR"]=="D") & (df["ps_prob"]>=-0.4)  & (df["ps_prob"]<-0.2)].index

ps5=df[(df["FTR"]=="D") & (df["ps_prob"]>=-0.2)  & (df["ps_prob"]<-2.22e-16)].index

ps6=df[(df["FTR"]=="D") & (df["ps_prob"]>=-2.22e-16)  & (df["ps_prob"]<0.2)].index

ps7=df[(df["FTR"]=="D") & (df["ps_prob"]>=0.2)  & (df["ps_prob"]<0.4)].index

ps8=df[(df["FTR"]=="D") & (df["ps_prob"]>=0.4)  & (df["ps_prob"]<0.6)].index

ps9=df[(df["FTR"]=="D") & (df["ps_prob"]>=0.6)  & (df["ps_prob"]<0.8)].index

ps10=df[(df["FTR"]=="D") & (df["ps_prob"]>=0.8)  & (df["ps_prob"]<1)].index

```


```python
ps_rows = [[len(ps1)],[len(ps2)],[len(ps3)],[len(ps4)],[(len(ps5))],[len(ps6)],[len(ps7)],[len(ps8)],[len(ps9)],[len(ps10)]]
```


```python
df_ps = pd.DataFrame(ps_rows, columns=["num_ps"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
ps_rng,bins=pd.cut((new_psh-new_psa),bins=bin_range,include_lowest=True, right=False,retbins=True)
```


```python
sorted_ps=ps_rng.value_counts(sort=True)
sorted_ps = sorted_ps.reindex(natsorted(sorted_ps.index))
```


```python
print(sorted_ps)
```

    [-1.0, -0.8)          1
    [-0.8, -0.6)         21
    [-0.6, -0.4)         26
    [-0.4, -0.2)         34
    [-0.2, -2.22e-16)    51
    [-2.22e-16, 0.2)     74
    [0.2, 0.4)           61
    [0.4, 0.6)           54
    [0.6, 0.8)           46
    [0.8, 1.0)           12
    dtype: int64
    


```python
df_sorted_ps = sorted_ps.to_frame("counts")
```


```python
df_ps.reset_index(drop=True, inplace=True)
df_sorted_ps.reset_index(drop=True, inplace=True)
```


```python
actl_prob_ps=df_ps["num_ps"]/df_sorted_ps["counts"]
```


```python
actl_prob_ps
```




    0    0.000000
    1    0.095238
    2    0.115385
    3    0.117647
    4    0.176471
    5    0.297297
    6    0.229508
    7    0.240741
    8    0.086957
    9    0.000000
    dtype: float64




```python
pointsps = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]
```


```python
ps_points = pd.DataFrame(pointsps, columns=["points_ps"])
```


```python
c=sns.scatterplot(x=new_psh-new_psa,y=new_psd)
d=sns.scatterplot(x=ps_points["points_ps"],y=actl_prob_ps)
c.set_xlabel("P(Home Win)-P(Away Win)")
c.set_ylabel("P(Draw)")
c.set_title("PS");
```


![png](output_88_0.png)



```python

```


```python
df["iw_prob"]=new_iwh-new_iwa
```


```python
df["iw_prob"]
```




    0      0.477850
    1      0.267454
    2      0.073378
    3     -0.458015
    4     -0.205885
             ...   
    375    0.621011
    376    0.638170
    377    0.534819
    378    0.198995
    379    0.120750
    Name: iw_prob, Length: 380, dtype: float64




```python
iw1=df[(df["FTR"]=="D") & (df["iw_prob"]>=-1)  & (df["iw_prob"]<-0.8)].index

iw2=df[(df["FTR"]=="D") & (df["iw_prob"]>=-0.8)  & (df["iw_prob"]<-0.6)].index

iw3=df[(df["FTR"]=="D") & (df["iw_prob"]>=-0.6)  & (df["iw_prob"]<-0.4)].index

iw4=df[(df["FTR"]=="D") & (df["iw_prob"]>=-0.4)  & (df["iw_prob"]<-0.2)].index

iw5=df[(df["FTR"]=="D") & (df["iw_prob"]>=-0.2)  & (df["iw_prob"]<-2.22e-16)].index

iw6=df[(df["FTR"]=="D") & (df["iw_prob"]>=-2.22e-16)  & (df["iw_prob"]<0.2)].index

iw7=df[(df["FTR"]=="D") & (df["iw_prob"]>=0.2)  & (df["iw_prob"]<0.4)].index

iw8=df[(df["FTR"]=="D") & (df["iw_prob"]>=0.4)  & (df["iw_prob"]<0.6)].index

iw9=df[(df["FTR"]=="D") & (df["iw_prob"]>=0.6)  & (df["iw_prob"]<0.8)].index

iw10=df[(df["FTR"]=="D") & (df["iw_prob"]>=0.8)  & (df["iw_prob"]<1)].index

```


```python
iw_rows = [[len(iw1)],[len(iw2)],[len(iw3)],[len(iw4)],[(len(iw5))],[len(iw6)],[len(iw7)],[len(iw8)],[len(iw9)],[len(iw10)]]
```


```python
df_iw = pd.DataFrame(iw_rows, columns=["num_iw"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
iw_rng,bins=pd.cut((new_iwh-new_iwa),bins=bin_range,include_lowest=True, right=False,retbins=True)
```


```python
sorted_iw=iw_rng.value_counts(sort=True)
sorted_iw = sorted_iw.reindex(natsorted(sorted_iw.index))
```


```python
print(sorted_iw)
```

    [-1.0, -0.8)          0
    [-0.8, -0.6)         17
    [-0.6, -0.4)         25
    [-0.4, -0.2)         38
    [-0.2, -2.22e-16)    50
    [-2.22e-16, 0.2)     83
    [0.2, 0.4)           68
    [0.4, 0.6)           48
    [0.6, 0.8)           46
    [0.8, 1.0)            5
    dtype: int64
    


```python
df_sorted_iw = sorted_iw.to_frame("counts")
```


```python
df_iw.reset_index(drop=True, inplace=True)
df_sorted_iw.reset_index(drop=True, inplace=True)
```


```python
actl_prob_iw=df_iw["num_iw"]/df_sorted_iw["counts"]
```


```python
actl_prob_iw
```




    0         NaN
    1    0.117647
    2    0.120000
    3    0.078947
    4    0.220000
    5    0.240964
    6    0.279412
    7    0.187500
    8    0.086957
    9    0.000000
    dtype: float64




```python
pointsiw = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]
```


```python
iw_points = pd.DataFrame(pointsiw, columns=["points_iw"])
```


```python
e=sns.scatterplot(x=new_iwh-new_iwa,y=new_iwd)
f=sns.scatterplot(x=iw_points["points_iw"],y=actl_prob_iw)
e.set_xlabel("P(Home Win)-P(Away Win)")
e.set_ylabel("P(Draw)")
e.set_title("IW");
```


![png](output_104_0.png)



```python

```


```python
df["bw_prob"]=new_bwh-new_bwa
```


```python
df["bw_prob"]
```




    0      0.501733
    1      0.285426
    2      0.065875
    3     -0.452785
    4     -0.225806
             ...   
    375    0.633311
    376    0.674740
    377    0.537953
    378    0.181818
    379    0.125993
    Name: bw_prob, Length: 380, dtype: float64




```python
bw1=df[(df["FTR"]=="D") & (df["bw_prob"]>=-1)  & (df["bw_prob"]<-0.8)].index

bw2=df[(df["FTR"]=="D") & (df["bw_prob"]>=-0.8)  & (df["bw_prob"]<-0.6)].index

bw3=df[(df["FTR"]=="D") & (df["bw_prob"]>=-0.6)  & (df["bw_prob"]<-0.4)].index

bw4=df[(df["FTR"]=="D") & (df["bw_prob"]>=-0.4)  & (df["bw_prob"]<-0.2)].index

bw5=df[(df["FTR"]=="D") & (df["bw_prob"]>=-0.2)  & (df["bw_prob"]<-2.22e-16)].index

bw6=df[(df["FTR"]=="D") & (df["bw_prob"]>=-2.22e-16)  & (df["bw_prob"]<0.2)].index

bw7=df[(df["FTR"]=="D") & (df["bw_prob"]>=0.2)  & (df["bw_prob"]<0.4)].index

bw8=df[(df["FTR"]=="D") & (df["bw_prob"]>=0.4)  & (df["bw_prob"]<0.6)].index

bw9=df[(df["FTR"]=="D") & (df["bw_prob"]>=0.6)  & (df["bw_prob"]<0.8)].index

bw10=df[(df["FTR"]=="D") & (df["bw_prob"]>=0.8)  & (df["bw_prob"]<1)].index

```


```python
bw_rows = [[len(bw1)],[len(bw2)],[len(bw3)],[len(bw4)],[(len(bw5))],[len(bw6)],[len(bw7)],[len(bw8)],[len(bw9)],[len(bw10)]]
```


```python
df_bw = pd.DataFrame(bw_rows, columns=["num_bw"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
bw_rng,bins=pd.cut((new_bwh-new_bwa),bins=bin_range,include_lowest=True, right=False,retbins=True)
```


```python
sorted_bw=bw_rng.value_counts(sort=True)
sorted_bw = sorted_bw.reindex(natsorted(sorted_bw.index))
```


```python
print(sorted_bw)
```

    [-1.0, -0.8)          0
    [-0.8, -0.6)         21
    [-0.6, -0.4)         26
    [-0.4, -0.2)         37
    [-0.2, -2.22e-16)    47
    [-2.22e-16, 0.2)     81
    [0.2, 0.4)           61
    [0.4, 0.6)           50
    [0.6, 0.8)           48
    [0.8, 1.0)            9
    dtype: int64
    


```python
df_sorted_bw = sorted_bw.to_frame("counts")
```


```python
df_bw.reset_index(drop=True, inplace=True)
df_sorted_bw.reset_index(drop=True, inplace=True)
```


```python
actl_prob_bw=df_bw["num_bw"]/df_sorted_bw["counts"]
```


```python
actl_prob_bw
```




    0         NaN
    1    0.095238
    2    0.115385
    3    0.108108
    4    0.191489
    5    0.271605
    6    0.262295
    7    0.220000
    8    0.083333
    9    0.000000
    dtype: float64




```python
pointsbw = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]
```


```python
bw_points = pd.DataFrame(pointsbw, columns=["points_bw"])
```


```python
g=sns.scatterplot(x=new_bwh-new_bwa,y=new_bwd)
h=sns.scatterplot(x=bw_points["points_bw"],y=actl_prob_bw)
g.set_xlabel("P(Home Win)-P(Away Win)")
g.set_ylabel("P(Draw)")
g.set_title("BW");
```


![png](output_120_0.png)



```python
# The blue points are boomakers prediction while orange points are the actual probabilities. 
#We can conclude that in the long run where orange point is above the blue one,it is likely to make some money for "Draw" in corresponding bin.
```


```python
#Task 3
```


```python
df["HR"].value_counts()
```




    0    362
    1     18
    Name: HR, dtype: int64




```python
df["AR"].value_counts()
```




    0    352
    1     27
    2      1
    Name: AR, dtype: int64




```python
df[(df["HR"]<1) & (df["AR"]<1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Div</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>FTR</th>
      <th>HTHG</th>
      <th>HTAG</th>
      <th>HTR</th>
      <th>...</th>
      <th>BbAvAHH</th>
      <th>BbMxAHA</th>
      <th>BbAvAHA</th>
      <th>PSCH</th>
      <th>PSCD</th>
      <th>PSCA</th>
      <th>wh_prob</th>
      <th>ps_prob</th>
      <th>iw_prob</th>
      <th>bw_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E0</td>
      <td>10/08/2018</td>
      <td>Man United</td>
      <td>Leicester</td>
      <td>2</td>
      <td>1</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>1.70</td>
      <td>2.29</td>
      <td>2.21</td>
      <td>1.55</td>
      <td>4.07</td>
      <td>7.69</td>
      <td>0.440842</td>
      <td>0.489448</td>
      <td>0.477850</td>
      <td>0.501733</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Bournemouth</td>
      <td>Cardiff</td>
      <td>2</td>
      <td>0</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>2.13</td>
      <td>1.80</td>
      <td>1.75</td>
      <td>1.88</td>
      <td>3.61</td>
      <td>4.70</td>
      <td>0.258252</td>
      <td>0.303796</td>
      <td>0.267454</td>
      <td>0.285426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Fulham</td>
      <td>Crystal Palace</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>2.11</td>
      <td>1.81</td>
      <td>1.77</td>
      <td>2.62</td>
      <td>3.38</td>
      <td>2.90</td>
      <td>0.047757</td>
      <td>0.065209</td>
      <td>0.073378</td>
      <td>0.065875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Huddersfield</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>3</td>
      <td>A</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>1.80</td>
      <td>2.13</td>
      <td>2.06</td>
      <td>7.24</td>
      <td>3.95</td>
      <td>1.58</td>
      <td>-0.435864</td>
      <td>-0.451328</td>
      <td>-0.458015</td>
      <td>-0.452785</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E0</td>
      <td>11/08/2018</td>
      <td>Newcastle</td>
      <td>Tottenham</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>1</td>
      <td>2</td>
      <td>A</td>
      <td>...</td>
      <td>2.12</td>
      <td>1.80</td>
      <td>1.76</td>
      <td>4.74</td>
      <td>3.53</td>
      <td>1.89</td>
      <td>-0.211241</td>
      <td>-0.214949</td>
      <td>-0.205885</td>
      <td>-0.225806</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>374</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Leicester</td>
      <td>Chelsea</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>D</td>
      <td>...</td>
      <td>2.38</td>
      <td>1.65</td>
      <td>1.62</td>
      <td>2.42</td>
      <td>3.63</td>
      <td>2.98</td>
      <td>0.069793</td>
      <td>0.066861</td>
      <td>0.076419</td>
      <td>0.070817</td>
    </tr>
    <tr>
      <th>375</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Liverpool</td>
      <td>Wolves</td>
      <td>2</td>
      <td>0</td>
      <td>H</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>1.91</td>
      <td>2.01</td>
      <td>1.95</td>
      <td>1.32</td>
      <td>5.89</td>
      <td>9.48</td>
      <td>0.652993</td>
      <td>0.648039</td>
      <td>0.621011</td>
      <td>0.633311</td>
    </tr>
    <tr>
      <th>376</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Man United</td>
      <td>Cardiff</td>
      <td>0</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>A</td>
      <td>...</td>
      <td>2.32</td>
      <td>1.72</td>
      <td>1.64</td>
      <td>1.30</td>
      <td>6.06</td>
      <td>9.71</td>
      <td>0.664234</td>
      <td>0.658818</td>
      <td>0.638170</td>
      <td>0.674740</td>
    </tr>
    <tr>
      <th>377</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Southampton</td>
      <td>Huddersfield</td>
      <td>1</td>
      <td>1</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>2.16</td>
      <td>1.80</td>
      <td>1.73</td>
      <td>1.37</td>
      <td>5.36</td>
      <td>8.49</td>
      <td>0.561325</td>
      <td>0.545368</td>
      <td>0.534819</td>
      <td>0.537953</td>
    </tr>
    <tr>
      <th>378</th>
      <td>E0</td>
      <td>12/05/2019</td>
      <td>Tottenham</td>
      <td>Everton</td>
      <td>2</td>
      <td>2</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>H</td>
      <td>...</td>
      <td>2.08</td>
      <td>1.85</td>
      <td>1.80</td>
      <td>1.91</td>
      <td>3.81</td>
      <td>4.15</td>
      <td>0.181818</td>
      <td>0.196429</td>
      <td>0.198995</td>
      <td>0.181818</td>
    </tr>
  </tbody>
</table>
<p>336 rows × 66 columns</p>
</div>




```python
removed_df=df[(df["HR"]<1) & (df["AR"]<1)]
```


```python
removed_df.shape  #First we have 380 matches and now we have 336 matches with no red card.
```




    (336, 66)




```python
#removed_BW
```


```python
rem_bwh=1/removed_df["BWH"]
```


```python
rem_bwh.head()
```




    0    0.653595
    1    0.526316
    2    0.408163
    3    0.160000
    4    0.263158
    Name: BWH, dtype: float64




```python
rem_bwd=1/removed_df["BWD"]
```


```python
rem_bwd.head()
```




    0    0.250000
    1    0.294118
    2    0.303030
    3    0.256410
    4    0.285714
    Name: BWD, dtype: float64




```python
rem_bwa=1/removed_df["BWA"]
rem_bwa.head()
```




    0    0.133333
    1    0.227273
    2    0.338983
    3    0.636943
    4    0.500000
    Name: BWA, dtype: float64




```python
#Calculating rem_BW values using Normalization
```


```python
rem_tbw=rem_bwh+rem_bwa+rem_bwd
```


```python
rem_tbw_=1/rem_tbw
```


```python
rem_new_bwh=rem_tbw_*rem_bwh
```


```python
rem_new_bwd=rem_tbw_*rem_bwd
```


```python
rem_new_bwa=rem_tbw_*rem_bwa
```


```python
#removed_IW
```


```python
rem_iwh=1/removed_df["IWH"]
rem_iwh.head()
```




    0    0.645161
    1    0.526316
    2    0.416667
    3    0.161290
    4    0.270270
    Name: IWH, dtype: float64




```python
rem_iwd=1/removed_df["IWD"]
rem_iwd.head()
```




    0    0.263158
    1    0.285714
    2    0.303030
    3    0.250000
    4    0.298507
    Name: IWD, dtype: float64




```python
rem_iwa=1/removed_df["IWA"]
rem_iwa.head()
```




    0    0.142857
    1    0.243902
    2    0.338983
    3    0.645161
    4    0.487805
    Name: IWA, dtype: float64




```python
#Calculating rem_IW values using Normalization
```


```python
rem_tiw=rem_iwh+rem_iwd+rem_iwa
```


```python
rem_tiw_=1/rem_tiw
```


```python
rem_new_iwh=rem_tiw_*rem_iwh
```


```python
rem_new_iwd=rem_tiw_*rem_iwd
```


```python
rem_new_iwa=rem_tiw_*rem_iwa
```


```python
#rem_PS
```


```python
rem_psh=1/removed_df["PSH"]
rem_psh.head()
```




    0    0.632911
    1    0.529101
    2    0.400000
    3    0.156006
    4    0.261097
    Name: PSH, dtype: float64




```python
rem_psd=1/removed_df["PSD"]
rem_psd.head()
```




    0    0.254453
    1    0.275482
    2    0.289017
    3    0.248756
    4    0.280112
    Name: PSD, dtype: float64




```python
rem_psa=1/removed_df["PSA"]
rem_psa.head()
```




    0    0.133333
    1    0.218341
    2    0.333333
    3    0.617284
    4    0.480769
    Name: PSA, dtype: float64




```python
#Calculating rem_PS values using Normalization
```


```python
rem_tps=rem_psh+rem_psd+rem_psa
```


```python
rem_tps_=1/rem_tps
```


```python
rem_new_psh=rem_tps_*rem_psh
```


```python
rem_new_psd=rem_tps_*rem_psd
```


```python
rem_new_psa=rem_tps_*rem_psa
```


```python
#rem_WH
```


```python
rem_whh=1/removed_df["WHH"]
rem_whh.head()
```




    0    0.636943
    1    0.523560
    2    0.408163
    3    0.172414
    4    0.263158
    Name: WHH, dtype: float64




```python
rem_whd=1/removed_df["WHD"]
rem_whd.head()
```




    0    0.263158
    1    0.285714
    2    0.303030
    3    0.256410
    4    0.312500
    Name: WHD, dtype: float64




```python
rem_wha=1/removed_df["WHA"]
rem_wha.head()
```




    0    0.166667
    1    0.250000
    2    0.357143
    3    0.636943
    4    0.487805
    Name: WHA, dtype: float64




```python
#Calculating rem_WH values using Normalization
```


```python
rem_twh=rem_whh+rem_wha+rem_whd
```


```python
rem_twh_=1/rem_twh
```


```python
rem_new_whh=rem_twh_*rem_whh
```


```python
rem_new_whd=rem_twh_*rem_whd
```


```python
rem_new_wha=rem_twh_*rem_wha
```


```python

```


```python

```


```python
df["rem_wh_prob"]=rem_new_whh-rem_new_wha
```


```python
df["rem_wh_prob"]
```




    0      0.440842
    1      0.258252
    2      0.047757
    3     -0.435864
    4     -0.211241
             ...   
    375    0.652993
    376    0.664234
    377    0.561325
    378    0.181818
    379         NaN
    Name: rem_wh_prob, Length: 380, dtype: float64




```python
rem_wh1=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=-1)  & (df["rem_wh_prob"]<-0.8)].index

rem_wh2=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=-0.8)  & (df["rem_wh_prob"]<-0.6)].index    

rem_wh3=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=-0.6)  & (df["rem_wh_prob"]<-0.4)].index

rem_wh4=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=-0.4)  & (df["rem_wh_prob"]<-0.2)].index

rem_wh5=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=-0.2)  & (df["rem_wh_prob"]<-2.22e-16)].index

rem_wh6=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=-2.22e-16)  & (df["rem_wh_prob"]<0.2)].index

rem_wh7=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=0.2)  & (df["rem_wh_prob"]<0.4)].index

rem_wh8=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=0.4)  & (df["rem_wh_prob"]<0.6)].index

rem_wh9=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=0.6)  & (df["rem_wh_prob"]<0.8)].index

rem_wh10=df[(df["FTR"]=="D") & (df["rem_wh_prob"]>=0.8)  & (df["rem_wh_prob"]<1)].index

```


```python
rem_wh_rows = [[len(rem_wh1)],[len(rem_wh2)],[len(rem_wh3)],[len(rem_wh4)],[(len(rem_wh5))],[len(rem_wh6)],[len(rem_wh7)],[len(rem_wh8)],[len(rem_wh9)],[len(rem_wh10)]]
```


```python
rem_df_wh = pd.DataFrame(rem_wh_rows, columns=["rem_num_wh"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
rem_wh_rng,bins=pd.cut((rem_new_whh-rem_new_wha),bins=bin_range,include_lowest=True, right=False,retbins=True)  #making bins
```


```python
rem_sorted_wh=rem_wh_rng.value_counts(sort=True)                         #sorting bins
rem_sorted_wh = rem_sorted_wh.reindex(natsorted(rem_sorted_wh.index))
```


```python
print(rem_sorted_wh)   
```

    [-1.0, -0.8)          1
    [-0.8, -0.6)         18
    [-0.6, -0.4)         21
    [-0.4, -0.2)         33
    [-0.2, -2.22e-16)    40
    [-2.22e-16, 0.2)     67
    [0.2, 0.4)           53
    [0.4, 0.6)           51
    [0.6, 0.8)           41
    [0.8, 1.0)           11
    dtype: int64
    


```python
rem_df_sorted_wh = rem_sorted_wh.to_frame("counts")
```


```python
rem_df_wh.reset_index(drop=True, inplace=True)
rem_df_sorted_wh.reset_index(drop=True, inplace=True)
```


```python
rem_actl_prob_wh=rem_df_wh["rem_num_wh"]/rem_df_sorted_wh["counts"]   #Number of games resulted "Draw" dividing by total number of games corresponding bin
```


```python
rem_actl_prob_wh
```




    0    0.000000
    1    0.111111
    2    0.095238
    3    0.151515
    4    0.175000
    5    0.283582
    6    0.245283
    7    0.215686
    8    0.097561
    9    0.000000
    dtype: float64




```python
rem_pointswh = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]  #the middle of intervals was chosen as point.
```


```python
rem_wh_points = pd.DataFrame(rem_pointswh, columns=["rem_points_wh"])
```


```python
j=sns.scatterplot(x=rem_new_whh-rem_new_wha,y=rem_new_whd)
k=sns.scatterplot(x=rem_wh_points["rem_points_wh"],y=rem_actl_prob_wh)
j.set_xlabel("P(Home Win)-P(Away Win)")
j.set_ylabel("P(Draw)")
j.set_title("rem_WH");
```


![png](output_186_0.png)



```python

```


```python
df["rem_ps_prob"]=rem_new_psh-rem_new_psa
```


```python
df["rem_ps_prob"]
```




    0      0.489448
    1      0.303796
    2      0.065209
    3     -0.451328
    4     -0.214949
             ...   
    375    0.648039
    376    0.658818
    377    0.545368
    378    0.196429
    379         NaN
    Name: rem_ps_prob, Length: 380, dtype: float64




```python
rem_ps1=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=-1)  & (df["rem_ps_prob"]<-0.8)].index

rem_ps2=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=-0.8)  & (df["rem_ps_prob"]<-0.6)].index    

rem_ps3=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=-0.6)  & (df["rem_ps_prob"]<-0.4)].index

rem_ps4=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=-0.4)  & (df["rem_ps_prob"]<-0.2)].index

rem_ps5=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=-0.2)  & (df["rem_ps_prob"]<-2.22e-16)].index

rem_ps6=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=-2.22e-16)  & (df["rem_ps_prob"]<0.2)].index

rem_ps7=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=0.2)  & (df["rem_ps_prob"]<0.4)].index

rem_ps8=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=0.4)  & (df["rem_ps_prob"]<0.6)].index

rem_ps9=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=0.6)  & (df["rem_ps_prob"]<0.8)].index

rem_ps10=df[(df["FTR"]=="D") & (df["rem_ps_prob"]>=0.8)  & (df["rem_ps_prob"]<1)].index

```


```python
rem_ps_rows = [[len(rem_ps1)],[len(rem_ps2)],[len(rem_ps3)],[len(rem_ps4)],[(len(rem_ps5))],[len(rem_ps6)],[len(rem_ps7)],[len(rem_ps8)],[len(rem_ps9)],[len(rem_ps10)]]
```


```python
rem_df_ps = pd.DataFrame(rem_ps_rows, columns=["rem_num_ps"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
rem_ps_rng,bins=pd.cut((rem_new_psh-rem_new_psa),bins=bin_range,include_lowest=True, right=False,retbins=True)  #making bins
```


```python
rem_sorted_ps=rem_ps_rng.value_counts(sort=True)                         #sorting bins
rem_sorted_ps = rem_sorted_ps.reindex(natsorted(rem_sorted_ps.index))
```


```python
print(rem_sorted_ps)   
```

    [-1.0, -0.8)          1
    [-0.8, -0.6)         19
    [-0.6, -0.4)         23
    [-0.4, -0.2)         30
    [-0.2, -2.22e-16)    43
    [-2.22e-16, 0.2)     58
    [0.2, 0.4)           55
    [0.4, 0.6)           54
    [0.6, 0.8)           41
    [0.8, 1.0)           12
    dtype: int64
    


```python
rem_df_sorted_ps = rem_sorted_ps.to_frame("counts")
```


```python
rem_df_ps.reset_index(drop=True, inplace=True)
rem_df_sorted_ps.reset_index(drop=True, inplace=True)
```


```python
rem_actl_prob_ps=rem_df_ps["rem_num_ps"]/rem_df_sorted_ps["counts"]   #Number of games resulted "Draw" dividing by total number of games corresponding bin
```


```python
rem_actl_prob_ps
```




    0    0.000000
    1    0.105263
    2    0.130435
    3    0.133333
    4    0.162791
    5    0.293103
    6    0.236364
    7    0.240741
    8    0.097561
    9    0.000000
    dtype: float64




```python
rem_pointsps = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]  #the middle of intervals was chosen as point.
```


```python
rem_ps_points = pd.DataFrame(rem_pointsps, columns=["rem_points_ps"])
```


```python

```


```python
l=sns.scatterplot(x=rem_new_psh-rem_new_psa,y=rem_new_psd)
m=sns.scatterplot(x=rem_ps_points["rem_points_ps"],y=rem_actl_prob_ps)
l.set_xlabel("P(Home Win)-P(Away Win)")
l.set_ylabel("P(Draw)")
l.set_title("rem_PS");
```


![png](output_203_0.png)



```python

```


```python
df["rem_iw_prob"]=rem_new_iwh-rem_new_iwa
```


```python
df["rem_iw_prob"]
```




    0      0.477850
    1      0.267454
    2      0.073378
    3     -0.458015
    4     -0.205885
             ...   
    375    0.621011
    376    0.638170
    377    0.534819
    378    0.198995
    379         NaN
    Name: rem_iw_prob, Length: 380, dtype: float64




```python
rem_iw1=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=-1)  & (df["rem_iw_prob"]<-0.8)].index

rem_iw2=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=-0.8)  & (df["rem_iw_prob"]<-0.6)].index    

rem_iw3=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=-0.6)  & (df["rem_iw_prob"]<-0.4)].index

rem_iw4=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=-0.4)  & (df["rem_iw_prob"]<-0.2)].index

rem_iw5=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=-0.2)  & (df["rem_iw_prob"]<-2.22e-16)].index

rem_iw6=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=-2.22e-16)  & (df["rem_iw_prob"]<0.2)].index

rem_iw7=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=0.2)  & (df["rem_iw_prob"]<0.4)].index

rem_iw8=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=0.4)  & (df["rem_iw_prob"]<0.6)].index

rem_iw9=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=0.6)  & (df["rem_iw_prob"]<0.8)].index

rem_iw10=df[(df["FTR"]=="D") & (df["rem_iw_prob"]>=0.8)  & (df["rem_iw_prob"]<1)].index

```


```python
rem_iw_rows = [[len(rem_iw1)],[len(rem_iw2)],[len(rem_iw3)],[len(rem_iw4)],[(len(rem_iw5))],[len(rem_iw6)],[len(rem_iw7)],[len(rem_iw8)],[len(rem_iw9)],[len(rem_iw10)]]
```


```python
rem_df_iw = pd.DataFrame(rem_iw_rows, columns=["rem_num_iw"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
rem_iw_rng,bins=pd.cut((rem_new_iwh-rem_new_iwa),bins=bin_range,include_lowest=True, right=False,retbins=True)  #making bins
```


```python
rem_sorted_iw=rem_iw_rng.value_counts(sort=True)                         #sorting bins
rem_sorted_iw = rem_sorted_iw.reindex(natsorted(rem_sorted_iw.index))
```


```python
print(rem_sorted_iw)   
```

    [-1.0, -0.8)          0
    [-0.8, -0.6)         17
    [-0.6, -0.4)         21
    [-0.4, -0.2)         33
    [-0.2, -2.22e-16)    44
    [-2.22e-16, 0.2)     65
    [0.2, 0.4)           62
    [0.4, 0.6)           48
    [0.6, 0.8)           41
    [0.8, 1.0)            5
    dtype: int64
    


```python
rem_df_sorted_iw = rem_sorted_iw.to_frame("counts")
```


```python
rem_df_iw.reset_index(drop=True, inplace=True)
rem_df_sorted_iw.reset_index(drop=True, inplace=True)
```


```python
rem_actl_prob_iw=rem_df_iw["rem_num_iw"]/rem_df_sorted_iw["counts"]   #Number of games resulted "Draw" dividing by total number of games corresponding bin
```


```python
rem_actl_prob_iw
```




    0         NaN
    1    0.117647
    2    0.142857
    3    0.090909
    4    0.204545
    5    0.246154
    6    0.274194
    7    0.187500
    8    0.097561
    9    0.000000
    dtype: float64




```python
rem_pointsiw = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]  #the middle of intervals was chosen as point.
```


```python
rem_iw_points = pd.DataFrame(rem_pointsiw, columns=["rem_points_iw"])
```


```python

```


```python
r=sns.scatterplot(x=rem_new_iwh-rem_new_iwa,y=rem_new_iwd)
s=sns.scatterplot(x=rem_iw_points["rem_points_iw"],y=rem_actl_prob_iw)
r.set_xlabel("P(Home Win)-P(Away Win)")
r.set_ylabel("P(Draw)")
r.set_title("rem_IW");
```


![png](output_220_0.png)



```python

```


```python
df["rem_bw_prob"]=rem_new_bwh-rem_new_bwa
```


```python
df["rem_bw_prob"]
```




    0      0.501733
    1      0.285426
    2      0.065875
    3     -0.452785
    4     -0.225806
             ...   
    375    0.633311
    376    0.674740
    377    0.537953
    378    0.181818
    379         NaN
    Name: rem_bw_prob, Length: 380, dtype: float64




```python
rem_bw1=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=-1)  & (df["rem_bw_prob"]<-0.8)].index

rem_bw2=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=-0.8)  & (df["rem_bw_prob"]<-0.6)].index    

rem_bw3=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=-0.6)  & (df["rem_bw_prob"]<-0.4)].index

rem_bw4=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=-0.4)  & (df["rem_bw_prob"]<-0.2)].index

rem_bw5=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=-0.2)  & (df["rem_bw_prob"]<-2.22e-16)].index

rem_bw6=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=-2.22e-16)  & (df["rem_bw_prob"]<0.2)].index

rem_bw7=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=0.2)  & (df["rem_bw_prob"]<0.4)].index

rem_bw8=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=0.4)  & (df["rem_bw_prob"]<0.6)].index

rem_bw9=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=0.6)  & (df["rem_bw_prob"]<0.8)].index

rem_bw10=df[(df["FTR"]=="D") & (df["rem_bw_prob"]>=0.8)  & (df["rem_bw_prob"]<1)].index

```


```python
rem_bw_rows = [[len(rem_bw1)],[len(rem_bw2)],[len(rem_bw3)],[len(rem_bw4)],[(len(rem_bw5))],[len(rem_bw6)],[len(rem_bw7)],[len(rem_bw8)],[len(rem_bw9)],[len(rem_bw10)]]
```


```python
rem_df_bw = pd.DataFrame(rem_bw_rows, columns=["rem_num_bw"])
```


```python
step = 0.2
bin_range = np.arange(-1, 1+step, step)
rem_bw_rng,bins=pd.cut((rem_new_bwh-rem_new_bwa),bins=bin_range,include_lowest=True, right=False,retbins=True)  #making bins
```


```python
rem_sorted_bw=rem_bw_rng.value_counts(sort=True)                         #sorting bins
rem_sorted_bw = rem_sorted_bw.reindex(natsorted(rem_sorted_iw.index))
```


```python
print(rem_sorted_bw)   
```

    [-1.0, -0.8)          0
    [-0.8, -0.6)         20
    [-0.6, -0.4)         22
    [-0.4, -0.2)         33
    [-0.2, -2.22e-16)    39
    [-2.22e-16, 0.2)     64
    [0.2, 0.4)           56
    [0.4, 0.6)           50
    [0.6, 0.8)           43
    [0.8, 1.0)            9
    dtype: int64
    


```python
rem_df_sorted_bw = rem_sorted_bw.to_frame("counts")
```


```python
rem_df_bw.reset_index(drop=True, inplace=True)
rem_df_sorted_bw.reset_index(drop=True, inplace=True)
```


```python
rem_actl_prob_bw=rem_df_bw["rem_num_bw"]/rem_df_sorted_bw["counts"]   #Number of games resulted "Draw" dividing by total number of games corresponding bin
```


```python
rem_actl_prob_bw
```




    0         NaN
    1    0.100000
    2    0.136364
    3    0.121212
    4    0.179487
    5    0.265625
    6    0.267857
    7    0.220000
    8    0.093023
    9    0.000000
    dtype: float64




```python
rem_pointsbw = [[-0.9],[-0.7],[-0.5],[-0.3],[-0.1],[0.1],[0.3],[0.5],[0.7],[0.9]]  #the middle of intervals was chosen as point.
```


```python
rem_bw_points = pd.DataFrame(rem_pointsbw, columns=["rem_points_bw"])
```


```python

```


```python
q=sns.scatterplot(x=rem_new_bwh-rem_new_bwa,y=rem_new_bwd)
w=sns.scatterplot(x=rem_bw_points["rem_points_bw"],y=rem_actl_prob_bw)
q.set_xlabel("P(Home Win)-P(Away Win)")
q.set_ylabel("P(Draw)")
q.set_title("rem_BW");
```


![png](output_237_0.png)



```python
# I would say that the red cards doesn't change my results significantly. Also, from previous analysis, it shows that while difference between P(Home)-P(Away) 
#approximates to zero, betting on draw is more likely to bring money.
```
