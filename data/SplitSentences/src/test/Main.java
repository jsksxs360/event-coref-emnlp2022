package test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Main {

	public static void main(String[] args) {
		
		String LDC2015E29 = "../LDC_TAC_KBP/LDC2015E29/data/source/mpdfxml/";
		String LDC2015E68 = "../LDC_TAC_KBP/LDC2015E68/data/source/";
		
		String KBP2015Train = "../LDC_TAC_KBP/LDC2017E02/data/2015/training/source/";
		String KBP2015Eval = "../LDC_TAC_KBP/LDC2017E02/data/2015/eval/source/";
		
		String KBP2016EvalNW = "../LDC_TAC_KBP/LDC2017E02/data/2016/eval/eng/nw/source/";
		String KBP2016EvalDF = "../LDC_TAC_KBP/LDC2017E02/data/2016/eval/eng/df/source/";
		
		String KBP2017EvalNW = "../LDC_TAC_KBP/LDC2017E54/data/eng/nw/source/";
		String KBP2017EvalDF = "../LDC_TAC_KBP/LDC2017E54/data/eng/df/source/";
		
		SourceReader reader = new SourceReader();
		List<Sentence> KBPSents = new LinkedList<>();
		try {
			// LDC2015E29
			List<Sentence> LDC2015E29Sents = reader.readSourceFolder(LDC2015E29);
			System.out.println("LDC2015E29: " + LDC2015E29Sents.size());
			KBPSents.addAll(LDC2015E29Sents);
			// LDC2015E68
			List<Sentence> LDC2015E68Sents = reader.readSourceFolder(LDC2015E68);
			System.out.println("LDC2015E68: " + LDC2015E68Sents.size());
			KBPSents.addAll(LDC2015E68Sents);
			// KBP2015
			List<Sentence> LDC2015TrainSents = reader.readSourceFolder(KBP2015Train);
			List<Sentence> LDC2015EvalSents = reader.readSourceFolder(KBP2015Eval);
			System.out.println("LDC2015: " + (LDC2015TrainSents.size() + LDC2015EvalSents.size()));
			KBPSents.addAll(LDC2015TrainSents);
			KBPSents.addAll(LDC2015EvalSents);
			// KBP 2016
			List<Sentence> KBP2016EvalNWSents = reader.readSourceFolder(KBP2016EvalNW, false);
			List<Sentence> KBP2016EvalDFSents = reader.readSourceFolder(KBP2016EvalDF, true);
			System.out.println("KBP2016: " + (KBP2016EvalNWSents.size() + KBP2016EvalDFSents.size()));
			KBPSents.addAll(KBP2016EvalNWSents);
			KBPSents.addAll(KBP2016EvalDFSents);
			// KBP 2017
			List<Sentence> KBP2017EvalNWSents = reader.readSourceFolder(KBP2017EvalNW, false);
			List<Sentence> KBP2017EvalDFSents = reader.readSourceFolder(KBP2017EvalDF, true);
			System.out.println("KBP2017: " + (KBP2017EvalNWSents.size() + KBP2017EvalDFSents.size()));
			KBPSents.addAll(KBP2017EvalNWSents);
			KBPSents.addAll(KBP2017EvalDFSents);
		} catch (Exception e) {
			e.printStackTrace();
		}
		try {
			saveFile("../kbp_sent.txt", KBPSents);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void saveFile(String filename, List<Sentence> sents) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
		for (Sentence sent : sents) {
			String text = sent.text.replace("\t", " ");
			if (isContainChinese(text) || text.startsWith("http") || text.startsWith("www.") || filter(text)) {
				continue;
			}
			writer.write(sent.filename + "\t" + sent.start + "\t" + text + "\n");
		}
		writer.close();
	}
	
	public static boolean filter(String str) {
		List<String> stopwords = Arrays.asList("P.S.", "PS", "snip", 
							"&amp;", "&lt;", "&gt;", "&nbsp;", "&quot;", 
							"#", "*", ".", "/", "year", "day", "month", "Â", "-", "[", "]",
							"!", "?", ",", ";", "(", ")", ":", "~", "_", 
							"cof", "sigh", "shrug", "and", "or", "done", "URL");
		for (String w : stopwords) {
			str = str.replace(w, " ");
		}
		Pattern p = Pattern.compile("[0-9]");
		Matcher matcher = p.matcher(str);
		str = matcher.replaceAll(" ");
		if (str.trim().isEmpty() || str.trim().length() == 1) return true;
		return false;
	}
	
	public static boolean isContainChinese(String str) {
		Pattern p = Pattern.compile("[\u4E00-\u9FA5]");
		Matcher m = p.matcher(str);
		if (m.find()) {
			return true;
		}
		return false;
	}

}
