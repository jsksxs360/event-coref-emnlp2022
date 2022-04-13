package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.StringUtils;

public class SourceReader {
	
	private StanfordCoreNLP pipeline;
	private List<String> newsStart =  Arrays.asList(new String[]{"AFP", "APW", "CNA", "NYT", "WPB", "XIN"});

	public SourceReader() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit");
		pipeline = new StanfordCoreNLP(props);
	}
	
	public List<Sentence> readSourceFolder(String folderPath, boolean isDF) throws Exception {
		File folder = new File(folderPath);
		List<Sentence> results = new LinkedList<>();
		for (String file : folder.list()) {
			results.addAll(readSourceFile(folderPath + file, isDF));
		}
		return results;
	}
	
	public List<Sentence> readSourceFolder(String folderPath) throws Exception {
		File folder = new File(folderPath);
		List<Sentence> results = new LinkedList<>();
		for (String file : folder.list()) {
			if (newsStart.contains(file.substring(0, 3))) { // News
				results.addAll(readSourceFile(folderPath + file, false));
			} else { // Forum
				results.addAll(readSourceFile(folderPath + file, true));
			}
		}
		return results;
	}
	
	public List<Sentence> readSourceFile(String filePath, boolean isDF) throws Exception {
		if (isDF) { // Forum
			return forumArticleReader(filePath, this.pipeline);
		} else { // News
			return newsArticleReader(filePath, this.pipeline);
		}
	}
	
	private static List<Sentence> forumArticleReader(String filePath, StanfordCoreNLP model) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(filePath));
		String filename = new File(filePath).getName();
		List<String> filters = Arrays.asList(".txt", ".xml", ".mpdf", ".cmp");
		for (String w : filters) {
			filename = filename.replace(w, "");
		}
		String line;
		int start = 0;
		List<Sentence> results = new LinkedList<>();
		while ((line = br.readLine()) != null) {
			int length = line.length() + 1;
			line = line.trim();
			if (line.startsWith("<DATE_TIME>") || line.startsWith("<DOC") || line.startsWith("<AUTHOR>")) {
				start += length;
				continue;
			}
			List<Sentence> sents = splitSentences(filename, line, start, model);
			results.addAll(sents);
			start += length;
		}
		br.close();
		return results;
	}
	
	private static List<Sentence> newsArticleReader(String filePath, StanfordCoreNLP model) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(filePath));
		String filename = new File(filePath).getName();
		List<String> filters = Arrays.asList(".txt", ".xml", ".mpdf", ".cmp");
		for (String w : filters) {
			filename = filename.replace(w, "");
		}
		String line;
		String Flag = "";
		String text = "";
		int start = 0;
		List<Sentence> results = new LinkedList<>();
		while ((line = br.readLine()) != null) {
			int length = line.length() + 1;
			if (line.trim().equals("<TEXT>")) {
				Flag = "TEXT";
				start += length;
				continue;
			} else if (line.trim().equals("<HEADLINE>") || line.trim().equals("<P>") || line.trim().equals("<KEYWORD>")) {
				Flag = "PARA";
				start += length;
				continue;
			} else if (line.trim().equals("</HEADLINE>") || line.trim().equals("</P>") || line.trim().equals("</KEYWORD>")) {
				Flag = "";
				List<Sentence> sentences = splitSentences(filename, text, start, model);
				results.addAll(sentences);
				start += text.length() + length;
				text = "";
				continue;
			} else if (line.trim().equals("</TEXT>")) {
				Flag = "";
				start += length;
				text = "";
				continue;
			}
			if (Flag.equals("PARA")) {
				text += line + " ";
				continue;
			} else if (Flag.equals("TEXT")) {
				List<Sentence> sentences = splitSentences(filename, line, start, model);
				results.addAll(sentences);
				start += length;
				continue;
			}
			start += length;
		}
		br.close();
		return results;
	}
	
	private static List<Sentence> splitSentences(String filename, String text, int start, StanfordCoreNLP pipeline) throws Exception {
		if (text.contains("<")) { // html file
			Pattern p_html = Pattern.compile("<[^>]+>", Pattern.CASE_INSENSITIVE);
			Matcher m_html = p_html.matcher(text);
			StringBuffer sb = new StringBuffer();
			while (m_html.find()) {
				m_html.appendReplacement(sb, StringUtils.repeat(" ", m_html.group().length()));
			}
			m_html.appendTail(sb);
			text = sb.toString();
			int count = 0;
			if (text.startsWith(" ")) {
				for (int i = 0; i < text.length(); i++) {
					if (text.charAt(i) != ' ') { break; }
					count += 1;
				}
			}
			text = text.trim();
			start += count;
		}
		// split sentence
		CoreDocument doc = new CoreDocument(text);
		pipeline.annotate(doc);
		List<Sentence> results = new LinkedList<>();
		for (CoreSentence sent : doc.sentences()) {
			Integer sentOffset = sent.charOffsets().first;
			String sentText = sent.text();
			if (sentText.isEmpty() || sentText.length() < 3) continue;
			results.add(new Sentence(filename, sentText, start + sentOffset));
		}
		return results;
	}
}
