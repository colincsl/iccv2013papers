# creates the nice .html page
# assumes that pdftowordcloud.py, pdftothumbs.py and scrape.py were already run

import os
import optparse
import json
import numpy as np

# CONFERENCE = 'CVPR'
# YEAR = 2012
# CONFERENCE = 'BMVC'
# YEAR = 2013
CONFERENCE = 'ICCV'
YEAR = 2013
BASE_FOLDER = os.path.expanduser("~/")+"Desktop/ConferenceGuides/"


def verifiy_folder_hierarchy(folder_name):
	folder_name = folder_name.split("/")
	for i in range(2,len(folder_name)):
		uri = "/".join(folder_name[:i])
		if not os.path.isdir(uri):
			os.mkdir(uri)
			print "Created folder:", uri

def generate_html(base_folder, conference, year):

	conf_folder = base_folder + "{}/{}/".format(year, conference)
	pdf_folder = conf_folder + "pdfs/"
	pdf_folder = conf_folder + "thumbs/"
	web_url = "{}web/{}/{}/".format(base_folder, year, conference) 
	verifiy_folder_hierarchy(web_url)	
	thumb_folder = web_url + 'thumbs/'

	# Load data
	all_papers = json.load(open(base_folder+"paper_names/"+conference+str(year)+"_papers.json"))
	public_count = sum([1 if x['PDF']!="" else 0 for x in all_papers])
	total_count = len(all_papers)
	data_json = json.load(open(web_url+"graph_data.json", 'r'))
	pairwise_str = str(data_json["pairwise"])
	pairwise = np.array(data_json["pairwise"])
	keywords = data_json["keywords"]
	# keywords = str([map(lambda x: str(x), x) for x in keywords])
	keywords = str(map(lambda x: str(x), keywords))

	# Create ordered matrix of size NxC where C is the # of classes
	groups = [int(x['group']) for x in data_json['nodes']]
	n_classes = len(np.unique(groups))
	loaddists = np.zeros([pairwise.shape[0], n_classes])
	for i,g in enumerate(groups):
		loaddists[i,g] = 1.

	pdf_prefix = 'http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/'

	# build up the string for html
	html = open("website_template.html", "r").read()
	s = ""
	all_papers = [x for x in all_papers if x['PDF'] is not u'']
	for i,paper in enumerate(all_papers):
		pid = i
		title = paper['Title'].encode('ascii', "ignore")
		authors = paper['Authors'].encode('ascii', "ignore")
		uri = paper['Title'] + " - " + paper['Authors'] + ".pdf"
		uri = uri.encode('ascii', errors='ignore')
		# print uri
		if paper['PDF'] != "":
			# pdflink = "http://"+paper['PDF']
			pdflink = pdf_prefix + paper['PDF']
		else:
			pdflink = """https://www.google.com/#q="{}" {}""".format(title, authors)

		s += """
		<div class="apaper" id="pid%d">
		<table><tr><td width=25px>
		<div class="circle circ%d"></div>
		</td><td>
		<div class="paperdesc">
			<span class="ts">%s <a href="%s">[pdf]</a> </span><br />
			<span class="as">%s</span>
		</div>
		</td></tr></table>
		<hr>
		""" % (pid, groups[i], title, pdflink, authors)

		# Try adding thumbnail
		# try:
		# 	if not os.path.exists(thumb_folder+uri+".jpg"):
		# 		raise ValueError
		# 	s += """<img src = "thumbs/%s.jpg"><br />""" % (uri.encode('ascii', errors='strict'))
		# except:
		# 	s += """<br />"""
		# 	# print 'a', uri

		# Setup table
		s += """<table class=paperTable><tr><td width=50% valign="valign">"""

		# Try adding abstract
		if "abstract" in data_json['nodes'][i].keys():
			abstract = data_json['nodes'][i]['abstract']
			# if abstract.find("Figure 1") > 0:
			# 	idx_end = abstract.find("Figure")
			# 	abstract = abstract[:idx_end]
			if len(abstract) > 1500:
				abstract = abstract[:1500]	
			# if title.find("Constant Time We") >= 0:
			# 		print title, abstract
			s += """
				<div class="abstract">
				<b>Abstract:</b> <i>{}</i>
				</div>
				""".format(abstract.encode("ascii", "ignore"))
			s += """</td><td width=50% valign="valign">"""



		# Add similar papers
		n_top_papers = 5
		idxs_top = pairwise[i].argsort(-1)[-n_top_papers-1:-1]
		s += """
			<div class="simpapers">
			<b>Similar papers:</b>	
			<ul>
			"""
		for j in idxs_top:
			s += """
				<li><b>{}</b> <a href={}>[pdf]</a> - <span class="as">{}</span> </li>
			""".format( all_papers[j]['Title'], pdf_prefix+all_papers[j]['PDF'], all_papers[j]['Authors'].encode('ascii', "ignore"))

		s += """
			</ul>
			</div>
			"""

		s += """
			<div class="dllinks">
				<span class="sim" id="sim%d">[rank all papers by similarity to this]</span><br /><br/>
			</div>
			""" % (pid)

		# End table
		s += "</td></tr></table>"
		# End section div
		s += """</div>"""

	# js += "]"
	# js2 += "]"

	# Replace elements in HTML template
	newhtml = html.replace("CONFERENCE", conference)
	newhtml = newhtml.replace("YEAR", str(year))
	newhtml = newhtml.replace("PUBLIC_COUNT", str(public_count))
	newhtml = newhtml.replace("TOTAL_COUNT", str(total_count))
	newhtml = newhtml.replace("RESULTTABLE", s)
	newhtml = newhtml.replace("LOADDISTS", str(loaddists.tolist()))
	newhtml = newhtml.replace("CLUSTERS", str(groups))
	newhtml = newhtml.replace("KEYWORDS", keywords)
	
	newhtml = newhtml.replace("PAIRDISTS", pairwise_str)

	index_url = web_url + "index.html"
	f = open(index_url, "w")
	f.write(newhtml)
	f.close()


	html = open("visualization_template.html", "r").read()
	# html = html.replace("CONFERENCE", conference)
	# html = html.replace("YEAR", str(year))
	viz_url = web_url + "visualization.html"
	f = open(viz_url, "w")
	f.write(html)
	f.close()


	# for pid, p in enumerate(paperdict):
	# 	# pid goes 1...N, p are the keys, pointing to actual paper IDs as given by NIPS, ~1...1500 with gaps

	# 	# get title, author
	# 	# print paperdict[p]
	# 	paper_id, title, author, filename, keywords = paperdict[p]
	# 	# print "Title:", title
	# 	# print "Author:", author
	# 	# print ""

	# 	# create the tags string
	# 	topwords = topdict.get(filename, [])
	# 	# embed()
	# 	# some top100 words may not have been computed during LDA so exclude them if
	# 	# they aren't found in wtoid
	# 	t = [x[0] for x in topwords if x[0] in wtoid]
	# 	tid = [int(argmax(phi[:, wtoid[x]])) for x in t] # assign each word to class
	# 	tcat = ""
	# 	for k in range(ldak):
	# 		ws = [x for i,x in enumerate(t) if tid[i]==k]
	# 		tcat += '[<span class="t'+ `k` + '">' + ", ".join(ws) + '</span>] '

	# 	# count up the complete distribution for the entire document and build up
	# 	# a javascript vector storing all this
	# 	svec = np.zeros(ldak)
	# 	for w in t:
	# 		svec += phi[:, wtoid[w]]
	# 	if svec.sum() == 0:
	# 		svec = ones(ldak)/ldak;
	# 	else:
	# 		svec = svec / svec.sum() # normalize
	# 	nums = [0 for k in range(ldak)]
	# 	for k in range(ldak):
	# 		nums[k] = "%.2f" % (float(svec[k]), )

	# 	js += "[" + ",".join(nums) + "]"
	# 	if not pid == len(paperdict)-1: js += ","

	# 	# dump similarities of this document to others
	# 	scores = ["%.2f" % (float(ds[pid, i]),) for i in range(N)]
	# 	js2 += "[" + ",".join(scores) + "]"
	# 	if not pid == len(paperdict)-1: js2 += ","

	# 	# get path to thumbnails for this paper
	# 	thumbpath = "thumbs/%s.jpg" % (filename, )

	# 	# get links to PDF, supplementary and bibtex on NIPS servers
	# 	pdflink = "http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/%s" % (filename, )
	# 	# bibtexlink = "http://books.nips.cc/papers/files/nips25/bibhtml/NIPS2012_%s.html" % (filename, )
	# 	# supplink = "http://books.nips.cc/papers/files/nips25/NIPS2012_%s.extra.zip" % (filename, )

	# 	s += """

	# 	<div class="apaper" id="pid%d">
	# 	<div class="paperdesc">
	# 		<span class="ts">%s</span><br />
	# 		<span class="as">%s</span><br />
	# 		<span class="keywords">Keywords: %s</span><br /><br />
	# 	</div>
	# 	<div class="dllinks">
	# 		<a href="%s">[pdf] </a>
	# 		<span class="sim" id="sim%d">[rank by tf-idf similarity to this]</span><br />
	# 		<!-- <span class="abstr" id="ab%d">[abstract]</span> -->
	# 	</div>
	# 	<img src = "%s"><br />
	# 	<!-- <div class = "abstrholder" id="abholder%d"></div> -->
	# 	<span class="tt">%s</span>
	# 	</div>

	# 	""" % (pid, title, author, keywords, pdflink, pid, int(p), thumbpath, int(p), tcat)


	# newhtml = html.replace("RESULTTABLE", s)

	# js += "]"
	# newhtml = newhtml.replace("LOADDISTS", js)

	# js2 += "]"
	# newhtml = newhtml.replace("PAIRDISTS", js2)

	# f = open("cvpr2013.html", "w")
	# f.write(newhtml)
	# f.close()


if __name__ == "__main__":

	parser = optparse.OptionParser()
	parser.add_option('-f', '--basefolder', dest='base_folder', default=BASE_FOLDER, help='base folder')
	parser.add_option('-c', '--conference', dest='conference', default=CONFERENCE, help='conference acronym')
	parser.add_option('-y', '--year', dest='year', default=YEAR, help='year')
	(opt, args) = parser.parse_args()

	generate_html(opt.base_folder, opt.conference, opt.year)

