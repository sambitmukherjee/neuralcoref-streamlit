# pip3 install -U spacy==2.1.0
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz --no-deps
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.1.0/en_core_web_md-2.1.0.tar.gz --no-deps
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/en_core_web_lg-2.1.0.tar.gz --no-deps
# pip3 install neuralcoref --no-binary neuralcoref
# pip3 install streamlit

import spacy
import en_core_web_md
import neuralcoref
import streamlit as st

@st.cache(allow_output_mutation=True)
def get_spacy_model(size="medium"):
	if size == "small":
		# nlp = spacy.load("en_core_web_sm")
		nlp = en_core_web_sm.load()
	elif size == "medium":
		# nlp = spacy.load("en_core_web_md")
		nlp = en_core_web_md.load()
	else:
		# nlp = spacy.load("en_core_web_lg")
		nlp = en_core_web_lg.load()
	neuralcoref.add_to_pipe(nlp)
	return nlp

@st.cache
def radio_format_func(raw_option):
	if raw_option == "example_paragraph":
		return "Select an example paragraph."
	else:
		return "Type your own paragraph."

st.title("Coreference Resolution Using NeuralCoref & spaCy")

chosen_mode = st.radio(
	label="Choose mode:",
	options=("example_paragraph", "own_paragraph"),
	format_func=radio_format_func,
	key="radio_key"
)

example_paragraphs = [
	"My sister has a dog. She loves him.",
	"Deepika too has a dog. The movie star has always been fond of animals.",
	"Sam has a Parker pen. He loves writing with it.",
	"Coronavirus quickly spread worldwide in 2020. The virus mostly affects elderly people. They can easily catch it.",
	"Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party.",
	"Jane told her friend that she was about to go to college.",
	"In 1916, a Polish American employee of Feltman's named Nathan Handwerker was encouraged by Eddie Cantor and Jimmy Durante, both working as waiters/musicians, to go into business in competition with his former employer. Handwerker undercut Feltman's by charging five cents for a hot dog when his former employer was charging ten.",
	"A dog named Teddy ran to his owner Jane. Jane loves her dog very much.",
	"Ana and Tom are siblings. Ana is older but her brother is taller.",
	"Angelica has three kittens. Her cats are very cute.",
	'"I like her", said Adam about Julia.',
	"Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger, with whom he shared an enthusiasm for computers. Paul and Bill used a teletype terminal at their high school, Lakeside, to develop their programming skills on several time-sharing computer systems.",
	"The legal pressures facing Michael Cohen are growing in a wide-ranging investigation of his personal business affairs and his work on behalf of his former client, President Trump. In addition to his work for Mr. Trump, he pursued his own business interests, including ventures in real estate, personal loans and investments in taxi medallions.",
	"We are looking for a region of central Italy bordering the Adriatic Sea. The area is mostly mountainous and includes Mt. Corno, the highest peak of the mountain range. It also includes many sheep and an Italian entrepreneur has an idea about how to make a little money of them."
]

with st.form(key="form_key"):
	if chosen_mode == "example_paragraph":
		paragraph = st.selectbox(
			label="Paragraph:",
			options=example_paragraphs,
			key="selectbox_key"
		)
	else:
		paragraph = st.text_area(
			label="Paragraph:",
			key="text_area_key"
		)
	submitted = st.form_submit_button(label="Resolve")
	if submitted:		
		nlp = get_spacy_model("medium")
		doc = nlp(paragraph)

		if doc._.has_coref:
			highlighted_original_text = ""
			for token in doc:
				if token._.in_coref:
					highlighted_original_text = highlighted_original_text + "<span style='color: #17B169;'>" + token.text + "</span>" + token.whitespace_
				else:
					highlighted_original_text = highlighted_original_text + token.text + token.whitespace_
			st.markdown(' > "' + highlighted_original_text + '"', unsafe_allow_html=True)
		else:
			st.markdown(' > "' + doc.text + '"', unsafe_allow_html=True)

		if doc._.has_coref:
			st.markdown("Any coreferences found? <span style='color: #17B169;'>YES</span>", unsafe_allow_html=True)
		else:
			st.markdown("Any coreferences found? <span style='color: #E94547;'>NO</span>", unsafe_allow_html=True)

		st.markdown("**Resolution:**")
		if doc._.has_coref:
			highlighted_resolved_text = doc._.coref_resolved
			for cluster in doc._.coref_clusters:
				highlighted_resolved_text = highlighted_resolved_text.replace(cluster.main.text, "<span style='color: #17B169;'>" + cluster.main.text + "</span>")
			st.markdown('> "' + highlighted_resolved_text + '"', unsafe_allow_html=True)
		else:
			st.markdown('> "' + doc._.coref_resolved + '"')
		
		if doc._.has_coref:
			st.markdown("**Clusters:**")
			st.write(doc._.coref_clusters)

"""
---

**References:**

1. [NeuralCoref](https://github.com/huggingface/neuralcoref)

"""
