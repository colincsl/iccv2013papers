iccv2013papers
==============


The code for this project is currently a mess. I plan on cleaning it up and generalizing it to other conferences in the future.

``r1_scrape_web.py``
This file searches the web for PDFs of all papers named in a json file (examples in /paper_names/). I suggest using [Beautiful Soup](http://www.crummy.com/software/BeautifulSoup/) to scrape paper names for conferences not listed. 

``r2_cluster_papers.py``
This extracts text from all papers listed in the json file and clusters papers based on the top 200 words in each document. Topics are formed using Kmeans on the pairwise TF-IDF scores. Other models such as LDA were tested but in practice produced worse results. The output of this program is another json file that has paper names/abstracts, topic labels, and pairwise scores between papers.

``r3_generate_html.py``
This uses the previous json file to create a webpage to sort/visualize the information. For an example see <http://colinlea.com/guides/2014/CVPR/>. To modify the website layout see ``website_template.html``

``r0_run_all.py``
Runs all of the scripts.
