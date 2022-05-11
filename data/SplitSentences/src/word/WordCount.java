package word;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import com.google.gson.Gson;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreEntityMention;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class WordCount {

	private StanfordCoreNLP pipeline;
	List<String> entityType = Arrays.asList("PERSON", "LOCATION", "ORGANIZATION");
	List<String> stopwords = Arrays.asList(
		"a", "an", "and", "are", "as", "at", "be", "but", "by",
		"for", "if", "in", "into", "is", "it", "been",
		"no", "not", "of", "on", "or", "such",
		"that", "the", "their", "then", "there", "these",
		"they", "this", "to", "was", "will", "with",
		"he", "she", "his", "her", "were", "do"
	);
	public WordCount() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
		props.setProperty("ner.applyFineGrained", "false");
		props.setProperty("ner.applyNumericClassifiers", "false");
		pipeline = new StanfordCoreNLP(props);
	}
	
	public Map<String, Integer> getVerbEntity(String document) {
		CoreDocument doc = new CoreDocument(document);
		Map<String, Integer> wordStatistic = new HashMap<String, Integer>();
		this.pipeline.annotate(doc);
		for (CoreLabel tok : doc.tokens()) {
			String word = tok.word().toLowerCase();
			if (this.stopwords.contains(word)) continue;
			if (tok.tag().startsWith("VB")) {
		    	wordStatistic.put(word, wordStatistic.getOrDefault(word, 0) + 1);
			}
	    }
	    for (CoreEntityMention em : doc.entityMentions()) {
	    	String entity = em.text().toLowerCase();
	    	if (this.stopwords.contains(entity) || !this.entityType.contains(em.entityType())) continue;
    		wordStatistic.put(entity, wordStatistic.getOrDefault(entity, 0) + 1);
	    }
	    return wordStatistic;
	}
	
	public static void main(String[] args) throws IOException {
		String kbp_sent_filePath = "../kbp_sent.txt";
		BufferedReader br = new BufferedReader(new FileReader(kbp_sent_filePath));
		String line;
		Map<String, String> kbp_documents = new HashMap<String, String>();
		while ((line = br.readLine()) != null) {
			String[] items = line.trim().split("\t");
			if (kbp_documents.containsKey(items[0])) {
				kbp_documents.replace(items[0], kbp_documents.get(items[0]) + " " + items[2]);
			} else {
				kbp_documents.put(items[0], items[2]);
			}
		}
		br.close();
		WordCount wc = new WordCount();
		Gson gson = new Gson();
		BufferedWriter bw = new BufferedWriter(new FileWriter("../kbp_word_count.txt"));
		for (Map.Entry<String, String> entry : kbp_documents.entrySet()) {
            System.out.println(entry.getKey());
			String countStr = gson.toJson(wc.getVerbEntity(entry.getValue()));
            bw.write(entry.getKey() + "\t" + countStr + "\n");
        }
		bw.close();
	}

}
