/* This file is part of the Z-MERT Training Tool for MT systems.
 * 
 * Z-MERT is an open-source tool, licensed under the terms of the
 * GNU Lesser General Public License (LGPL). Therefore, it is free
 * for personal and scientific use by individuals and/or research
 * groups. It may not be modified or redistributed, publicly or
 * privately, unless the licensing terms are observed. If in doubt,
 * contact the author for clarification and/or an explicit
 * permission.
 *
 * If you use Z-MERT in your work, please cite the following paper:
 *
 *       Omar F. Zaidan. 2009. Z-MERT: A Fully Configurable Open
 *       Source Tool for Minimum Error Rate Training of Machine
 *       Translation Systems. The Prague Bulletin of Mathematical
 *       Linguistics, No. 91:79-88.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307 USA
 */

import java.util.*;
import java.text.DecimalFormat;

// BUG: try using joshua.util.io.LineReader instead
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;

public class EvalTool {
	final static DecimalFormat f4 = new DecimalFormat("###0.0000");
	
	// if true, evaluation is performed for each candidate translation as
	// well as on the entire candidate set
	static boolean verbose;
	
	// number of sentences in the dev set
	static int numSentences;
	
	// number of documents in the dev set
	static int numDocuments;

	// docOfSentence[i] stores which document contains the i'th sentence.
	// docOfSentence is 0-indexed, as are the documents (i.e. first doc is indexed 0)
	static int[] docOfSentence;
	
	// names of documents
	static String[] docNames;
	
	// stores information regarding which subset of the documents are evaluated
	// [0]: method (0-6)
	// [1]: first (1-indexed)
	// [2]: last (1-indexed)
	// [3]: size
	// [4]: center
	// [5]: arg1
	// [6]: arg2
	// [1-6] are 0 for method 0, [6] is 0 for methods 1-4 as well
	// only [1] and [2] are needed for optimization. The rest are only needed for an output message.
	static int[] docSubsetInfo;

	// number of reference translations per sentence
	static int refsPerSen;
	
	// 0: no normalization, 1: "NIST-style" tokenization, and also rejoin 'm, 're, *'s, 've, 'll, 'd, and n't,
	// 2: apply 1 and also rejoin dashes between letters, 3: apply 1 and also drop non-ASCII characters
	// 4: apply 1+2+3
	static private int textNormMethod;
	
	// refSentences[i][r] is the rth reference translation of the ith sentence
	static String[][] refSentences;
	
	// name of evaluation metric
	static String metricName;

	// name of evaluation metric optimized by MERT, possibly with "doc-level " prefixed
	static String metricName_display;

	// options for the evaluation metric (e.g. for BLEU, maxGramLength and effLengthMethod)
	static String[] metricOptions;
	
	// the scorer
	static EvaluationMetric evalMetric;
	
	// if true, the reference set(s) is (are) evaluated
	static boolean evaluateRefs;
	
	// file informing Z-MERT which document each sentence belongs to
	static String docInfoFileName;
	
	// file names for input files.  When refsPerSen > 1, refFileName can be
	// the name of a single file, or a file name prefix.
	static String refFileName;
	static String candFileName;
	
	// format of the candidate file: "plain" if one candidate per sentence, and "nbest" if a decoder output
	static String candFileFormat;
	
	// if format is nbest, evaluate the r'th candidate of each sentence
	static int candRank;

	
	private static void evaluateCands_plain(String inFileName) {
		evaluate(candFileName, "plain", 1, 1);
	}
	
	
	private static void evaluateCands_nbest(String inFileName, int testIndex) {
		evaluate(candFileName, "nbest", -1, testIndex);
	}
	
	
	private static void evaluateRefSet(int r) {
		evaluate(refFileName, "plain", refsPerSen, r);
	}
	
	
	private static void evaluate(String inFileName, String inFileFormat, int candPerSen, int testIndex) {
		// candPerSen: how many candidates are provided per sentence?
		//             (if inFileFormat is nbest, then candPerSen is ignored, since it is variable)
		// testIndex: which of the candidates (for each sentence) should be tested?
		//            e.g. testIndex=1 means first candidate should be evaluated
		//                 testIndex=candPerSen means last candidate should be evaluated
		
		if (inFileFormat.equals("plain") && candPerSen < 1) {
			println("candPerSen must be positive for a file in plain format.");
			System.exit(30);
		}
		
		if (inFileFormat.equals("plain") && (testIndex < 1 || testIndex > candPerSen)) {
			println("For the plain format, testIndex must be in [1,candPerSen]");
			System.exit(31);
		}
		

		String[] topCand_str = new String[numSentences];
		
		// BUG: all of this needs to be replaced with the SegmentFileParser and related interfaces.
		try {
			
			// read the candidates
			
			InputStream inStream = new FileInputStream(new File(inFileName));
			BufferedReader inFile = new BufferedReader(new InputStreamReader(inStream, "utf8"));
			String line, candidate_str;
			
			if (inFileFormat.equals("plain")) {

				for (int i = 0; i < numSentences; ++i) {

					// skip candidates 1 through testIndex-1
					for (int n = 1; n < testIndex; ++n) {
						line = inFile.readLine();
					}
					
					// read testIndex'th candidate
					candidate_str = inFile.readLine();
					
					topCand_str[i] = normalize(candidate_str, textNormMethod);
					
					for (int n = testIndex+1; n <= candPerSen; ++n){
						// skip candidates testIndex+1 through candPerSen-1
						// (this probably only applies when evaluating a combined reference file)
						line = inFile.readLine();
					}

				} // for (i)
				
			} else { // nbest format
				
				int i = 0;
				int n = 1;
				line = inFile.readLine();
				
				while (line != null && i < numSentences) {

/*
line format:

.* ||| words of candidate translation . ||| feat-1_val feat-2_val ... feat-numParams_val .*

*/
					
					while (n < candRank) {
						line = inFile.readLine();
						++n;
					}
					
					// at the moment, line stores the candRank'th candidate (1-indexed) of the i'th sentence (0-indexed)
					
					if (line == null) {
						println("Not enough candidates in " + inFileName + " to extract the " + candRank + "'th candidate for each sentence.");
						println("(Failed to extract one for the " + i + "'th sentence (0-indexed).)");
						System.exit(32);
					}
					
					int read_i = Integer.parseInt(line.substring(0,line.indexOf(" |||")).trim());
					if (read_i == i) {
						line = line.substring(line.indexOf("||| ")+4); // get rid of initial text
						candidate_str = line.substring(0,line.indexOf(" |||"));
						topCand_str[i] = normalize(candidate_str, textNormMethod);
						if (i < numSentences-1) {
							while (read_i == i) {
								line = inFile.readLine();
								read_i = Integer.parseInt(line.substring(0,line.indexOf(" |||")).trim());
							}
						}
						n = 1;
						i += 1;
					} else {
						println("Not enough candidates in " + inFileName + " to extract the " + candRank + "'th candidate for each sentence.");
						println("(Failed to extract one for the " + i + "'th sentence (0-indexed).)");
						System.exit(32);
					}
					
				} // while (line != null)
				
				if (i != numSentences) {
					println("Not enough candidates were found (i = " + i + "; was expecting " + numSentences + ")");
					System.exit(33);
				}

			} // nbest format
			
			inFile.close();
			
		} catch (FileNotFoundException e) {
			System.err.println("FileNotFoundException in EvalTool.initialize(int): " + e.getMessage());
			System.exit(99901);
		} catch (IOException e) {
			System.err.println("IOException in EvalTool.initialize(int): " + e.getMessage());
			System.exit(99902);
		}
		
		
		int[] IA = new int[numSentences];
		for (int i = 0; i < numSentences; ++i) { IA[i] = i; }
		int[][] SS = evalMetric.suffStats(topCand_str,IA);
		
		int suffStatsCount = evalMetric.get_suffStatsCount();
		
		int[][] totStats_doc = new int[numDocuments][suffStatsCount];
		for (int doc = 0; doc < numDocuments; ++doc) {
			for (int s = 0; s < suffStatsCount; ++s) {
				totStats_doc[doc][s] = 0;
			}
		}
		
		for (int i = 0; i < numSentences; ++i) {
			int docOf_i = docOfSentence[i];
			for (int s = 0; s < suffStatsCount; ++s) {
				totStats_doc[docOf_i][s] += SS[i][s];
			}
		}
		
		int[] totStats_corpus = new int[suffStatsCount];
		for (int s = 0; s < suffStatsCount; ++s) {
			totStats_corpus[s] = 0;
			for (int doc = 0; doc < numDocuments; ++doc) {
				totStats_corpus[s] += totStats_doc[doc][s];
			}
		}
		
		if (verbose) {
			println("");
			println("Printing detailed scores for individual sentences...");
			for (int i = 0; i < numSentences; ++i) {
				print("Sentence #" + i + ": ");
				int[] stats = new int[suffStatsCount];
				for (int s = 0; s < suffStatsCount; ++s) { stats[s] = SS[i][s]; }
				evalMetric.printDetailedScore_fromStats(stats,true);
				// already prints a \n
			}
		}
		
		if (numDocuments > 1) {
			println("");
			println("Document level individual scores:");
			evalMetric.printDetailedScore_fromStats(totStats_doc,docNames);
			println("  => Document level average score: " + f4.format(evalMetric.score(totStats_doc)));
		}
		
		println("");
		println("Corpus level score:");
		evalMetric.printDetailedScore_fromStats(totStats_corpus,false);
		
	} // void evaluate(...)
	
		
	private static void printUsage(int argsLen) {
		println("Oops, you provided " + argsLen + " args!");
		println("");
		println("Usage:");
		println(" EvalTool [-cand candFile] [-format candFileformat] [-rank r]\n            [-ref refFile] [-rps refsPerSen] [-m metricName metric options]\n            [-evr evalRefs] [-v verbose]");
		println("");
		println(" (*) -cand candFile: candidate translations\n       [[default: candidates.txt]]");
		println(" (*) -format candFileFormat: is the candidate file a plain file (one candidate\n       per sentence) or does it contain multiple candidates per sentence as\n       a decoder's output)?  For the first, use \"plain\".  For the second,\n       use \"nbest\".\n       [[default: plain]]");
		println(" (*) -rank r: if format=nbest, evaluate the set of r'th candidates.\n       [[default: 1]]");
		println(" (*) -ref refFile: reference translations (or file name prefix)\n       [[default: references.txt]]");
		println(" (*) -rps refsPerSen: number of reference translations per sentence\n       [[default: 1]]");
		println(" (*) -txtNrm textNormMethod: how should text be normalized?\n          (0) don't normalize text,\n       or (1) \"NIST-style\", and also rejoin 're, *'s, n't, etc,\n       or (2) apply 1 and also rejoin dashes between letters,\n       or (3) apply 1 and also drop non-ASCII characters,\n       or (4) apply 1+2+3\n       [[default: 1]]");
		println(" (*) -docInfo documentInfoFile: file informing Z-MERT which document each\n          sentence belongs to\n       [[default: null string (i.e. all sentences are in one 'document')]]");
		println(" (*) -m metricName metric options: name of evaluation metric and its options\n       [[default: BLEU 4 closest]]");
		println(" (*) -evr evalRefs: evaluate references (1) or not (0) (sanity check)\n       [[default: 0]]");
		println(" (*) -v verbose: evaluate individual sentences (1) or not (0)\n       [[default: 0]]");
		println("");
		println("Ex.: java EvalTool -cand nbest.out -ref ref.all -rps 4 -m BLEU 4 shortest");
	}
	
	
	private static void processArgsAndInitialize(String[] args) {
		EvaluationMetric.set_knownMetrics();
		
		// set default values
		candFileName = "candidates.txt";
		candFileFormat = "plain";
		candRank = 1;
		refFileName = "references.txt";
		refsPerSen = 1;
		textNormMethod = 1;
		metricName = "BLEU";
		metricOptions = new String[2];
		metricOptions[0] = "4";
		metricOptions[1] = "closest";
		docSubsetInfo = new int[7];
		docSubsetInfo[0] = 0;
		evaluateRefs = false;
		verbose = false;
		
		int i = 0;
		
		while (i < args.length) {
			String option = args[i];
			if (option.equals("-cand")) {
				candFileName = args[i+1];
			} else if (option.equals("-format")) {
				candFileFormat = args[i+1];
				if (!candFileFormat.equals("plain") && !candFileFormat.equals("nbest")) {
					println("candFileFormat must be either plain or nbest.");
					System.exit(10);
				}
			} else if (option.equals("-rank")) {
				candRank = Integer.parseInt(args[i+1]);
				if (refsPerSen < 1) {
					println("Argument for -rank must be positive.");
					System.exit(10);
				}
			} else if (option.equals("-ref")) {
				refFileName = args[i+1];
			} else if (option.equals("-rps")) {
				refsPerSen = Integer.parseInt(args[i+1]);
				if (refsPerSen < 1) {
					println("refsPerSen must be positive.");
					System.exit(10);
				}
			} else if (option.equals("-txtNrm")) {
				textNormMethod = Integer.parseInt(args[i+1]);
				if (textNormMethod < 0 || textNormMethod > 4) {
					println("textNormMethod should be between 0 and 4");
					System.exit(10);
				}
			} else if (option.equals("-docInfo")) {
				docInfoFileName = args[i+1];
			} else if (option.equals("-m")) {
				metricName = args[i+1];
				if (EvaluationMetric.knownMetricName(metricName)) {
					int optionCount = EvaluationMetric.metricOptionCount(metricName);
					metricOptions = new String[optionCount];
					for (int opt = 0; opt < optionCount; ++opt) {
						metricOptions[opt] = args[i+opt+2];
					}
					i += optionCount;
				} else {
					println("Unknown metric name " + metricName + ".");
					System.exit(10);
				}
			} else if (option.equals("-docSet")) {
				String method = args[i+1];
				
				if (method.equals("all")) {
					docSubsetInfo[0] = 0;
					i += 0;
				} else if (method.equals("bottom")) {
					String a = args[i+2];
					if (a.endsWith("d")) {
						docSubsetInfo[0] = 1;
						a = a.substring(0,a.indexOf("d"));
					} else {
						docSubsetInfo[0] = 2;
						a = a.substring(0,a.indexOf("%"));
					}
					docSubsetInfo[5] = Integer.parseInt(a);
					i += 1;
				} else if (method.equals("top")) {
					String a = args[i+2];
					if (a.endsWith("d")) {
						docSubsetInfo[0] = 3;
						a = a.substring(0,a.indexOf("d"));
					} else {
						docSubsetInfo[0] = 4;
						a = a.substring(0,a.indexOf("%"));
					}
					docSubsetInfo[5] = Integer.parseInt(a);
					i += 1;
				} else if (method.equals("window")) {
					String a1 = args[i+2];
					a1 = a1.substring(0,a1.indexOf("d")); // size of window
					String a2 = args[i+4];
					if (a2.indexOf("p") > 0) {
						docSubsetInfo[0] = 5;
						a2 = a2.substring(0,a2.indexOf("p"));
					} else {
						docSubsetInfo[0] = 6;
						a2 = a2.substring(0,a2.indexOf("r"));
					}
					docSubsetInfo[5] = Integer.parseInt(a1);
					docSubsetInfo[6] = Integer.parseInt(a2);
					i += 3;
				} else {
					println("Unknown docSet method " + method + ".");
					System.exit(10);
				}
			} else if (option.equals("-evr")) {
				int evr = Integer.parseInt(args[i+1]);
				if (evr == 1) {
					evaluateRefs = true;
				} else if (evr == 0) {
					evaluateRefs = false;
				} else {
					println("evalRefs must be either 0 or 1.");
					System.exit(10);
				}
			} else if (option.equals("-v")) {
				int v = Integer.parseInt(args[i+1]);
				if (v == 1) {
					verbose = true;
				} else if (v == 0) {
					verbose = false;
				} else {
					println("verbose must be either 0 or 1.");
					System.exit(10);
				}
			} else {
				println("Unknown option " + option); System.exit(10);
			}
			
			i += 2;
			
		} // while (i)
		
		if (refsPerSen > 1) {
			// the provided refFileName might be a prefix
			File dummy = new File(refFileName);
			if (!dummy.exists()) {
				refFileName = createUnifiedRefFile(refFileName,refsPerSen);
			}
		} else {
			checkFile(refFileName);
		}
		
		
		// initialize
		numSentences = countLines(refFileName) / refsPerSen;
		
		// sets numDocuments and docOfSentence[]
		processDocInfo();
		
		if (numDocuments > 1) metricName_display = "doc-level " + metricName;
		
		set_docSubsetInfo(docSubsetInfo);
		
		
		
		// read in reference sentences
		refSentences = new String[numSentences][refsPerSen];
		
		try {
			
			InputStream inStream_refs = new FileInputStream(new File(refFileName));
			BufferedReader inFile_refs = new BufferedReader(new InputStreamReader(inStream_refs, "utf8"));
			
			for (i = 0; i < numSentences; ++i) {
				for (int r = 0; r < refsPerSen; ++r) {
					// read the rth reference translation for the ith sentence
					refSentences[i][r] = normalize(inFile_refs.readLine(), textNormMethod);
				}
			}
			
			inFile_refs.close();
			
		} catch (FileNotFoundException e) {
			System.err.println("FileNotFoundException in EvalTool.initialize(int): " + e.getMessage());
			System.exit(99901);
		} catch (IOException e) {
			System.err.println("IOException in EvalTool.initialize(int): " + e.getMessage());
			System.exit(99902);
		}
		
		// set static data members for the EvaluationMetric class
		EvaluationMetric.set_numSentences(numSentences);
		EvaluationMetric.set_numDocuments(numDocuments);
		EvaluationMetric.set_refsPerSen(refsPerSen);
		EvaluationMetric.set_refSentences(refSentences);
		
		// do necessary initialization for the evaluation metric
		evalMetric = EvaluationMetric.getMetric(metricName,metricOptions);
		
		println("Processing " + numSentences + " sentences...");
		
	} // processArgsAndInitialize(String[] args)

  private static void processDocInfo()
  {
    // sets numDocuments and docOfSentence[]
    docOfSentence = new int[numSentences];

    if (docInfoFileName == null) {
      for (int i = 0; i < numSentences; ++i) docOfSentence[i] = 0;
      numDocuments = 1;
    } else {

      try {

        // 4 possible formats:
        //   1) List of numbers, one per document, indicating # sentences in each document.
        //   2) List of "docName size" pairs, one per document, indicating name of document and # sentences.
        //   3) List of docName's, one per sentence, indicating which doument each sentence belongs to.
        //   4) List of docName_number's, one per sentence, indicating which doument each sentence belongs to,
        //      and its order in that document. (can also use '-' instead of '_')

        int docInfoSize = countNonEmptyLines(docInfoFileName);

        if (docInfoSize < numSentences) { // format #1 or #2
          numDocuments = docInfoSize;
          int i = 0;

          BufferedReader inFile = new BufferedReader(new FileReader(docInfoFileName));
          String line = inFile.readLine();
          boolean format1 = (!(line.contains(" ")));

          for (int doc = 0; doc < numDocuments; ++doc) {

            if (doc != 0) line = inFile.readLine();

            int docSize = 0;
            if (format1) {
              docSize = Integer.parseInt(line);
            } else {
              docSize = Integer.parseInt(line.split("\\s+")[1]);
            }

            for (int i2 = 1; i2 <= docSize; ++i2) {
              docOfSentence[i] = doc;
              ++i;
            }

          }

          // now i == numSentences

          inFile.close();

        } else if (docInfoSize == numSentences) { // format #3 or #4

          boolean format3 = false;

          HashSet<String> seenStrings = new HashSet<String>();
          BufferedReader inFile = new BufferedReader(new FileReader(docInfoFileName));
          for (int i = 0; i < numSentences; ++i) {
            // set format3 = true if a duplicate is found
            String line = inFile.readLine();
            if (seenStrings.contains(line)) format3 = true;
            seenStrings.add(line);
          }

          inFile.close();

          HashSet<String> seenDocNames = new HashSet<String>();
          HashMap<String,Integer> docOrder = new HashMap<String,Integer>();
            // maps a document name to the order (0-indexed) in which it was seen

          inFile = new BufferedReader(new FileReader(docInfoFileName));
          for (int i = 0; i < numSentences; ++i) {
            String line = inFile.readLine();

            String docName = "";
            if (format3) {
              docName = line;
            } else {
              int sep_i = Math.max(line.lastIndexOf('_'),line.lastIndexOf('-'));
              docName = line.substring(0,sep_i);
            }

            if (!seenDocNames.contains(docName)) {
              seenDocNames.add(docName);
              docOrder.put(docName,seenDocNames.size()-1);
            }

            int docOrder_i = docOrder.get(docName);

            docOfSentence[i] = docOrder_i;

          }

          inFile.close();

          numDocuments = seenDocNames.size();

        } else { // badly formatted

        }

      } catch (FileNotFoundException e) {
        System.err.println("FileNotFoundException in EvalTool.processDocInfo(): " + e.getMessage());
        System.exit(99901);
      } catch (IOException e) {
        System.err.println("IOException in EvalTool.processDocInfo(): " + e.getMessage());
        System.exit(99902);
      }
    }

  }


  private static void set_docSubsetInfo(int[] info)
  {

/*
1: -docSet bottom 8d
2: -docSet bottom 25%				the bottom ceil(0.20*numDocs) documents
3: -docSet top 8d
4: -docSet top 25%					the top ceil(0.20*numDocs) documents

5: -docSet window 11d around 90percentile		11 docs centered around 80th percentile
												(complain if not enough docs; don't adjust)
6: -docSet window 11d around 40rank				11 docs centered around doc ranked 50
		                						(complain if not enough docs; don't adjust)


[0]: method (0-6)
[1]: first (1-indexed)
[2]: last (1-indexed)
[3]: size
[4]: center
[5]: arg1 (-1 for method 0)
[6]: arg2 (-1 for methods 0-4)
*/
    if (info[0] == 0) { // all
      info[1] = 1;
      info[2] = numDocuments;
      info[3] = numDocuments;
      info[4] = (info[1] + info[2]) / 2;
    } if (info[0] == 1) { // bottom d
      info[3] = info[5];
      info[2] = numDocuments;
      info[1] = numDocuments - info[3] + 1;
      info[4] = (info[1] + info[2]) / 2;
    } if (info[0] == 2) { // bottom p
      info[3] = (int)(Math.ceil((info[5]/100.0) * numDocuments));
      info[2] = numDocuments;
      info[1] = numDocuments - info[3] + 1;
      info[4] = (info[1] + info[2]) / 2;
    } if (info[0] == 3) { // top d
      info[3] = info[5];
      info[1] = 1;
      info[2] = info[3];
      info[4] = (info[1] + info[2]) / 2;
    } if (info[0] == 4) { // top p
      info[3] = (int)(Math.ceil((info[5]/100.0) * numDocuments));
      info[1] = 1;
      info[2] = info[3];
      info[4] = (info[1] + info[2]) / 2;
    } if (info[0] == 5) { // window around percentile
      info[3] = info[5];
      info[4] = (int)(Math.floor((info[6]/100.0) * numDocuments));
      info[1] = info[4] - ((info[3]-1) / 2);
      info[2] = info[4] + ((info[3]-1) / 2);
    } if (info[0] == 6) { // window around rank
      info[3] = info[5];
      info[4] = info[6];
      info[1] = info[4] - ((info[3]-1) / 2);
      info[2] = info[4] + ((info[3]-1) / 2);
    }

  }

	private static void checkFile(String fileName) {
		if (!fileExists(fileName)) {
			println("The file " + fileName + " was not found!");
			System.exit(40);
		}
	}
	
	
	private static boolean fileExists(String fileName) {
		File checker = new File(fileName);
		return checker.exists();
	}
	
	
	private static String createUnifiedRefFile(String prefix, int numFiles) {
		if (numFiles < 2) {
			println("Warning: createUnifiedRefFile called with numFiles = " + numFiles + "; doing nothing.");
			return prefix;
		} else {
			File checker;
			checker = new File(prefix + "1");
			
			if (!checker.exists()) {
				checker = new File(prefix + ".1");
				if (!checker.exists()) {
					println("Can't find reference files.");
					System.exit(50);
				} else {
					prefix = prefix + ".";
				}
			}
			
			String outFileName;
			if (prefix.endsWith(".")) {
				outFileName = prefix + "all";
			} else {
				outFileName = prefix + ".all";
			}
			
			try {
				PrintWriter outFile = new PrintWriter(outFileName);
				
				BufferedReader[] inFile = new BufferedReader[numFiles];
				
				int nextIndex;
				checker = new File(prefix + "0");
				if (checker.exists()) {
					nextIndex = 0;
				} else {
					nextIndex = 1;
				}
				int lineCount = countLines(prefix + nextIndex);
				
				for (int r = 0; r < numFiles; ++r) {
					if (countLines(prefix + nextIndex) != lineCount) {
						println("Line count mismatch in " + (prefix+nextIndex) + ".");
						System.exit(60);
					}
					InputStream inStream = new FileInputStream(new File(prefix + nextIndex));
					inFile[r] = new BufferedReader(new InputStreamReader(inStream, "utf8"));
					++nextIndex;
				}
				
				String line;
				
				for (int i = 0; i < lineCount; ++i) {
					for (int r = 0; r < numFiles; ++r) {
						line = inFile[r].readLine();
						outFile.println(line);
					}
				}
				
				outFile.close();
				
				for (int r = 0; r < numFiles; ++r) {
					inFile[r].close();
				}
				
			} catch (FileNotFoundException e) {
				System.err.println("FileNotFoundException in EvalTool.createUnifiedRefFile(String,int): " + e.getMessage());
				System.exit(99901);
			} catch (IOException e) {
				System.err.println("IOException in EvalTool.createUnifiedRefFile(String,int): " + e.getMessage());
				System.exit(99902);
			}
			
			return outFileName;
			
		}
		
	} // createUnifiedRefFile(String prefix, int numFiles)
	
	private static String normalize(String str, int normMethod)
	{
		if (normMethod == 0) return str;
		
		// replace HTML/SGML
		str = str.replaceAll("&quot;","\"");
		str = str.replaceAll("&amp;","&");
		str = str.replaceAll("&lt;","<");
		str = str.replaceAll("&gt;",">");
		str = str.replaceAll("&apos;","'");
		
		
		
		// split on these characters:
		// ! " # $ % & ( ) * + / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
		// i.e. ASCII 33-126, except alphanumeric, and except "," "-" "." "'"
		
		//                 ! "#  $%&  (  )  *  +/:;<=>  ?@  [   \  ]  ^_`  {  |  }~
		String split_on = "!\"#\\$%&\\(\\)\\*\\+/:;<=>\\?@\\[\\\\\\]\\^_`\\{\\|\\}~";
		
//		println("split_on: " + split_on);
		
		for (int k = 0; k < split_on.length(); ++k) {
			// for each split character, reprocess the string
			String regex = "" + split_on.charAt(k);
			if (regex.equals("\\")) {
				++k;
				regex += split_on.charAt(k);
			}
			str = str.replaceAll(regex," " + regex + " ");
		}
		
		
		
		// split on "." and "," and "-", conditioned on proper context
		
		str = " " + str + " ";
		str = str.replaceAll("\\s+"," ");
		
		TreeSet<Integer> splitIndices = new TreeSet<Integer>();
		
		for (int i = 0; i < str.length(); ++i) {
			char ch = str.charAt(i);
			if (ch == '.' || ch == ',') {
				// split if either of the previous or next characters is a non-digit
				char prev_ch = str.charAt(i-1);
				char next_ch = str.charAt(i+1);
				if (prev_ch < '0' || prev_ch > '9' || next_ch < '0' || next_ch > '9') {
					splitIndices.add(i);
				}
			} else if (ch == '-') {
				// split if preceded by a digit
				char prev_ch = str.charAt(i-1);
				if (prev_ch >= '0' && prev_ch <= '9') {
					splitIndices.add(i);
				}
			}
		}
		
		String str0 = str;
		str = "";
		
		for (int i = 0; i < str0.length(); ++i) {
			if (splitIndices.contains(i)) {
				str += " " + str0.charAt(i) + " ";
			} else {
				str += str0.charAt(i);
			}
		}
		
		
		
		// rejoin i'm, we're, *'s, won't, don't, etc
		
		str = " " + str + " ";
		str = str.replaceAll("\\s+"," ");
		
		str = str.replaceAll(" i 'm "," i'm ");
		str = str.replaceAll(" we 're "," we're ");
		str = str.replaceAll(" 's ","'s ");
		str = str.replaceAll(" 've ","'ve ");
		str = str.replaceAll(" 'll ","'ll ");
		str = str.replaceAll(" 'd ","'d ");
		str = str.replaceAll(" n't ","n't ");
		
		
		
		// remove spaces around dashes
		if (normMethod == 2 || normMethod == 4) {
			
			TreeSet<Integer> skipIndices = new TreeSet<Integer>();
			str = " " + str + " ";
			
			for (int i = 0; i < str.length(); ++i) {
				char ch = str.charAt(i);
				if (ch == '-') {
					// rejoin if surrounded by spaces, and then letters
					if (str.charAt(i-1) == ' ' && str.charAt(i+1) == ' ') {
						if (Character.isLetter(str.charAt(i-2)) && Character.isLetter(str.charAt(i+2))) {
							skipIndices.add(i-1);
							skipIndices.add(i+1);
						}
					}
				}
			}
			
			str0 = str;
			str = "";
			
			for (int i = 0; i < str0.length(); ++i) {
				if (!skipIndices.contains(i)) {
					str += str0.charAt(i);
				}
			}
		}
		
		
		
		// drop non-ASCII characters
		if (normMethod == 3 || normMethod == 4) {
			
			str0 = str;
			str = "";
			
			for (int i = 0; i < str0.length(); ++i) {
				char ch = str0.charAt(i);
				if (ch <= 127) { // i.e. if ASCII
					str += ch;
				}
			}
		}
		
		
		
		str = str.replaceAll("\\s+"," ");
		
		str = str.trim();
		
		return str;
	}
	
	// TODO: we should handle errors properly for the three use sites of this function, and should remove the function.
	//       OK, but we don't want it to use LineReader, so it can function within the standalone release of Z-MERT. -- O.Z.
	private static int countLines(String fileName) {
		int count = 0;
		
		try {
			BufferedReader inFile = new BufferedReader(new FileReader(fileName));
			
			String line;
			do {
				line = inFile.readLine();
				if (line != null) ++count;
			}  while (line != null);
			
			inFile.close();
		} catch (IOException e) {
			System.err.println("IOException in EvalTool.countLines(String): " + e.getMessage());
			System.exit(99902);
		}
		
		return count;
	}

	private static int countNonEmptyLines(String fileName)
	{
		int count = 0;
		
		try {
			BufferedReader inFile = new BufferedReader(new FileReader(fileName));
			
			String line;
			do {
				line = inFile.readLine();
				if (line != null && line.length() > 0) ++count;
			}  while (line != null);
			
			inFile.close();
		} catch (IOException e) {
			System.err.println("IOException in EvalTool.countNonEmptyLines(String): " + e.getMessage());
			System.exit(99902);
		}
		
		return count;
	}
	
	private static void println(Object obj) { System.out.println(obj); }
	private static void print(Object obj) { System.out.print(obj); }

	public static void main(String[] args) {
		if (args.length == 0) {
			printUsage(args.length);
			System.exit(0);
		} else {
			processArgsAndInitialize(args);
		}
		// non-specified args will be set to default values in processArgsAndInitialize
		
		if (candFileFormat.equals("plain")) {
			println("Evaluating candidate translations in plain file " + candFileName + "...");
			evaluateCands_plain(candFileName);
		} else if (candFileFormat.equals("nbest")) {
			println("Evaluating set of " + candRank + "'th candidate translations from " + candFileName + "...");
			evaluateCands_nbest(candFileName,candRank);
		}
		println("");
		
		if (evaluateRefs) {
			// evaluate the references themselves; useful if developing a new evaluation metric
			
			println("");
			println("PERFORMING SANITY CHECK:");
			println("------------------------");
			println("");
			println("This metric's scores range from "
				+ evalMetric.worstPossibleScore() + " (worst) to "
				+ evalMetric.bestPossibleScore() + " (best).");
			
			for (int r = 1; r <= refsPerSen; ++r) {
				println("");
				println("(*) Evaluating reference set " + r + ":");
				println("");
				evaluateRefSet(r);
				println("");
			}
		}
		
//		System.exit(0);
		
	} // main(String[] args)

}
