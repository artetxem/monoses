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
import java.util.logging.Logger;

public class BLEU extends EvaluationMetric
{
	private static final Logger logger = Logger.getLogger(BLEU.class.getName());
	
  protected int maxGramLength;
  protected EffectiveLengthMethod effLengthMethod;
    // 1: closest, 2: shortest, 3: average
//  protected HashMap[][] maxNgramCounts;
  protected HashMap<String,Integer>[] maxNgramCounts;
  protected int[][] refWordCount;
  protected double[] weights;

  public BLEU()
  {
    this(4,"closest");
  }

  public BLEU(String[] BLEU_options)
  {
    this(Integer.parseInt(BLEU_options[0]),BLEU_options[1]);
  }

  public BLEU(int mxGrmLn,String methodStr)
  {
    if (mxGrmLn >= 1) {
      maxGramLength = mxGrmLn;
    } else {
      logger.severe("Maximum gram length must be positive");
      System.exit(1);
    }

    if (methodStr.equals("closest")) {
      effLengthMethod = EffectiveLengthMethod.CLOSEST;
    } else if (methodStr.equals("shortest")) {
      effLengthMethod = EffectiveLengthMethod.SHORTEST;
//    } else if (methodStr.equals("average")) {
//      effLengthMethod = EffectiveLengthMethod.AVERAGE;
    } else {
    	logger.severe("Unknown effective length method string " + methodStr + ".");
//      System.out.println("Should be one of closest, shortest, or average.");
    	logger.severe("Should be one of closest or shortest.");
      System.exit(1);
    }

    initialize();

  }



  protected void initialize()
  {
    metricName = "BLEU";
    toBeMinimized = false;
    suffStatsCount = 2*maxGramLength + 2;
      // 2 per gram length for its precision, and 2 for length info
    set_weightsArray();
    set_maxNgramCounts();
  }

  public double bestPossibleScore() { return 1.0; }
  public double worstPossibleScore() { return 0.0; }

  protected void set_weightsArray()
  {
    weights = new double[1+maxGramLength];
    for (int n = 1; n <= maxGramLength; ++n) {
      weights[n] = 1.0/maxGramLength;
    }
  }


  protected void set_maxNgramCounts()
  {
    @SuppressWarnings("unchecked")
    HashMap<String,Integer>[] temp_HMA = new HashMap[numSentences];
    maxNgramCounts = temp_HMA;

    String gram = "";
    int oldCount = 0, nextCount = 0;

      for (int i = 0; i < numSentences; ++i) {
        maxNgramCounts[i] = getNgramCountsAll(refSentences[i][0]);
          // initialize to ngramCounts[n] of the first reference translation...

        // ...and update as necessary from the other reference translations
        for (int r = 1; r < refsPerSen; ++r) {
          HashMap<String,Integer> nextNgramCounts = getNgramCountsAll(refSentences[i][r]);
          Iterator<String> it = (nextNgramCounts.keySet()).iterator();

          while (it.hasNext()) {
            gram = it.next();
            nextCount = nextNgramCounts.get(gram);

            if (maxNgramCounts[i].containsKey(gram)) { // update if necessary
              oldCount = maxNgramCounts[i].get(gram);
              if (nextCount > oldCount) {
                maxNgramCounts[i].put(gram,nextCount);
              }
            } else { // add it
              maxNgramCounts[i].put(gram,nextCount);
            }

          }

        } // for (r)

      } // for (i)

    // For efficiency, calculate the reference lenghts, which will be used in effLength...

    refWordCount = new int[numSentences][refsPerSen];
    for (int i = 0; i < numSentences; ++i) {
      for (int r = 0; r < refsPerSen; ++r) {
        refWordCount[i][r] = wordCount(refSentences[i][r]);
      }
    }

  }


  public int[] suffStats(String cand_str, int i)
  {
    int[] stats = new int[suffStatsCount];

//int wordCount = words.length;
//for (int j = 0; j < wordCount; ++j) { words[j] = words[j].intern(); }

    if (!cand_str.equals("")) {
      String[] words = cand_str.split("\\s+");
      set_prec_suffStats(stats,words,i);
      stats[suffStatsCount-2] = words.length;
      stats[suffStatsCount-1] = effLength(words.length,i);
    } else {
      String[] words = new String[0];
      set_prec_suffStats(stats,words,i);
      stats[suffStatsCount-2] = 0;
      stats[suffStatsCount-1] = effLength(0,i);
    }

    return stats;
  }

  public void set_prec_suffStats(int[] stats, String[] words, int i)
  {
    HashMap<String,Integer>[] candCountsArray = getNgramCountsArray(words);

    for (int n = 1; n <= maxGramLength; ++n) {

      int correctGramCount = 0;
      String gram = "";
      int candGramCount = 0, maxRefGramCount = 0, clippedCount = 0;

      Iterator<String> it = (candCountsArray[n].keySet()).iterator();

      while (it.hasNext()) {
      // for each n-gram type in the candidate
        gram = it.next();
        candGramCount = candCountsArray[n].get(gram);
//        if (maxNgramCounts[i][n].containsKey(gram)) {
//          maxRefGramCount = maxNgramCounts[i][n].get(gram);
        if (maxNgramCounts[i].containsKey(gram)) {
          maxRefGramCount = maxNgramCounts[i].get(gram);
        } else {
          maxRefGramCount = 0;
        }

        clippedCount = Math.min(candGramCount,maxRefGramCount);

        correctGramCount += clippedCount;

      }

      stats[2*(n-1)] = correctGramCount;
      stats[2*(n-1)+1] = Math.max(words.length-(n-1),0); // total gram count

    } // for (n)

  }

  public int effLength(int candLength, int i)
  {
    if (effLengthMethod == EffectiveLengthMethod.CLOSEST) { // closest

      int closestRefLength = refWordCount[i][0];
      int minDiff = Math.abs(candLength-closestRefLength);

      for (int r = 1; r < refsPerSen; ++r) {
        int nextRefLength = refWordCount[i][r];
        int nextDiff = Math.abs(candLength-nextRefLength);

        if (nextDiff < minDiff) {
          closestRefLength = nextRefLength;
          minDiff = nextDiff;
        } else if (nextDiff == minDiff && nextRefLength < closestRefLength) {
          closestRefLength = nextRefLength;
          minDiff = nextDiff;
        }
      }

      return closestRefLength;

    } else if (effLengthMethod == EffectiveLengthMethod.SHORTEST) { // shortest

      int shortestRefLength = refWordCount[i][0];

      for (int r = 1; r < refsPerSen; ++r) {
        int nextRefLength = refWordCount[i][r];
        if (nextRefLength < shortestRefLength) {
          shortestRefLength = nextRefLength;
        }
      }

      return shortestRefLength;

    }
/* // commented out because it needs sufficient statistics to be doubles
else { // average

      int totalRefLength = refWordCount[i][0];

      for (int r = 1; r < refsPerSen; ++r) {
        totalRefLength += refWordCount[i][r];
      }

      return totalRefLength/(double)refsPerSen;

    }
*/
    return candLength; // should never get here anyway

  }

  public double score(int[] stats)
  {
    if (stats.length != suffStatsCount) {
      logger.severe("Mismatch between stats.length and suffStatsCount (" + stats.length + " vs. " + suffStatsCount + ") in BLEU.score(int[])");
      System.exit(2);
    }

    double BLEUsum = 0.0;
    double smooth_addition = 1.0; // following bleu-1.04.pl
    double c_len = stats[suffStatsCount-2];
    double r_len = stats[suffStatsCount-1];

    double correctGramCount, totalGramCount;

    for (int n = 1; n <= maxGramLength; ++n) {
      correctGramCount = stats[2*(n-1)];
      totalGramCount = stats[2*(n-1)+1];

      double prec_n;
      if (totalGramCount > 0) {
        prec_n = correctGramCount/totalGramCount;
      } else {
        prec_n = 1; // following bleu-1.04.pl ???????
      }

      if (prec_n == 0) {
        smooth_addition *= 0.5;
        prec_n = smooth_addition / (c_len-n+1);
        // isn't c_len-n+1 just totalGramCount ???????
      }

      BLEUsum += weights[n] * Math.log(prec_n);

    }

    double BP = 1.0;
    if (c_len < r_len) BP = Math.exp(1-(r_len/c_len));
      // if c_len > r_len, no penalty applies

    return BP*Math.exp(BLEUsum);

  }

  public void printDetailedScore_fromStats(int[] stats, boolean oneLiner)
  {
    double BLEUsum = 0.0;
    double smooth_addition = 1.0; // following bleu-1.04.pl
    double c_len = stats[suffStatsCount-2];
    double r_len = stats[suffStatsCount-1];

    double correctGramCount, totalGramCount;

    if (oneLiner) {
      System.out.print("Precisions: ");
    }

    for (int n = 1; n <= maxGramLength; ++n) {
      correctGramCount = stats[2*(n-1)];
      totalGramCount = stats[2*(n-1)+1];

      double prec_n;
      if (totalGramCount > 0) {
        prec_n = correctGramCount/totalGramCount;
      } else {
        prec_n = 1; // following bleu-1.04.pl ???????
      }

      if (prec_n > 0) {
        if (totalGramCount > 0) {
          if (oneLiner) {
            System.out.print(n + "=" + f4.format(prec_n) + ", ");
          } else {
            System.out.println("BLEU_precision(" + n + ") = " + (int)correctGramCount + " / " + (int)totalGramCount + " = " + f4.format(prec_n));
          }
        } else {
          if (oneLiner) {
            System.out.print(n + "=N/A, ");
          } else {
            System.out.println("BLEU_precision(" + n + ") = N/A (candidate has no " + n + "-grams)");
          }
        }
      } else {
        smooth_addition *= 0.5;
        prec_n = smooth_addition / (c_len-n+1);
        // isn't c_len-n+1 just totalGramCount ???????

        if (oneLiner) {
          System.out.print(n + "~" + f4.format(prec_n) + ", ");
        } else {
          System.out.println("BLEU_precision(" + n + ") = " + (int)correctGramCount + " / " + (int)totalGramCount + " ==smoothed==> " + f4.format(prec_n));
        }
      }

      BLEUsum += weights[n] * Math.log(prec_n);

    }

    if (oneLiner) {
      System.out.print("(overall=" + f4.format(Math.exp(BLEUsum)) + "), ");
    } else {
      System.out.println("BLEU_precision = " + f4.format(Math.exp(BLEUsum)));
      System.out.println("");
    }

    double BP = 1.0;
    if (c_len < r_len) BP = Math.exp(1-(r_len/c_len));
      // if c_len > r_len, no penalty applies

    if (oneLiner) {
      System.out.print("BP=" + f4.format(BP) + ", ");
    } else {
      System.out.println("Length of candidate corpus = " + (int)c_len);
      System.out.println("Effective length of reference corpus = " + (int)r_len);
      System.out.println("BLEU_BP = " + f4.format(BP));
      System.out.println("");
    }

    System.out.println("  => BLEU = " + f4.format(BP*Math.exp(BLEUsum)));
  }


  protected int wordCount(String cand_str)
  {
    if (!cand_str.equals("")) {
      return cand_str.split("\\s+").length;
    } else {
      return 0;
    }
  }


  public HashMap<String,Integer>[] getNgramCountsArray(String cand_str)
  {
    if (!cand_str.equals("")) {
      return getNgramCountsArray(cand_str.split("\\s+"));
    } else {
      return getNgramCountsArray(new String[0]);
    }
  }


  public HashMap<String,Integer>[] getNgramCountsArray(String[] words)
  {
    @SuppressWarnings("unchecked")
    HashMap<String,Integer>[] ngramCountsArray = new HashMap[1+maxGramLength];
    ngramCountsArray[0] = null;
    for (int n = 1; n <= maxGramLength; ++n) {
      ngramCountsArray[n] = new HashMap<String,Integer>();
    }

    int len = words.length;
    String gram;
    int st = 0;

    for (; st <= len-maxGramLength; ++st) {

      gram = words[st];
      if (ngramCountsArray[1].containsKey(gram)) {
        int oldCount = ngramCountsArray[1].get(gram);
        ngramCountsArray[1].put(gram,oldCount+1);
      } else {
        ngramCountsArray[1].put(gram,1);
      }

      for (int n = 2; n <= maxGramLength; ++n) {
        gram = gram + " " + words[st+n-1];
        if (ngramCountsArray[n].containsKey(gram)) {
          int oldCount = ngramCountsArray[n].get(gram);
          ngramCountsArray[n].put(gram,oldCount+1);
        } else {
          ngramCountsArray[n].put(gram,1);
        }
      } // for (n)

    } // for (st)

    // now st is either len-maxGramLength+1 or zero (if above loop never entered, which
    // happens with sentences that have fewer than maxGramLength words)

    for (; st < len; ++st) {

      gram = words[st];
      if (ngramCountsArray[1].containsKey(gram)) {
        int oldCount = ngramCountsArray[1].get(gram);
        ngramCountsArray[1].put(gram,oldCount+1);
      } else {
        ngramCountsArray[1].put(gram,1);
      }

      int n = 2;
      for (int fin = st+1; fin < len; ++fin) {
        gram = gram + " " + words[st+n-1];

        if (ngramCountsArray[n].containsKey(gram)) {
          int oldCount = ngramCountsArray[n].get(gram);
          ngramCountsArray[n].put(gram,oldCount+1);
        } else {
          ngramCountsArray[n].put(gram,1);
        }
        ++n;
      } // for (fin)

    } // for (st)

    return ngramCountsArray;

  }


  public HashMap<String,Integer> getNgramCountsAll(String cand_str)
  {
    if (!cand_str.equals("")) {
      return getNgramCountsAll(cand_str.split("\\s+"));
    } else {
      return getNgramCountsAll(new String[0]);
    }
  }

  public HashMap<String,Integer> getNgramCountsAll(String[] words)
  {
    HashMap<String,Integer> ngramCountsAll = new HashMap<String,Integer>();

    int len = words.length;
    String gram;
    int st = 0;

    for (; st <= len-maxGramLength; ++st) {

      gram = words[st];
      if (ngramCountsAll.containsKey(gram)) {
        int oldCount = ngramCountsAll.get(gram);
        ngramCountsAll.put(gram,oldCount+1);
      } else {
        ngramCountsAll.put(gram,1);
      }

      for (int n = 2; n <= maxGramLength; ++n) {
        gram = gram + " " + words[st+n-1];
        if (ngramCountsAll.containsKey(gram)) {
          int oldCount = ngramCountsAll.get(gram);
          ngramCountsAll.put(gram,oldCount+1);
        } else {
          ngramCountsAll.put(gram,1);
        }
      } // for (n)

    } // for (st)

    // now st is either len-maxGramLength+1 or zero (if above loop never entered, which
    // happens with sentences that have fewer than maxGramLength words)

    for (; st < len; ++st) {

      gram = words[st];
      if (ngramCountsAll.containsKey(gram)) {
        int oldCount = ngramCountsAll.get(gram);
        ngramCountsAll.put(gram,oldCount+1);
      } else {
        ngramCountsAll.put(gram,1);
      }

      int n = 2;
      for (int fin = st+1; fin < len; ++fin) {
        gram = gram + " " + words[st+n-1];

        if (ngramCountsAll.containsKey(gram)) {
          int oldCount = ngramCountsAll.get(gram);
          ngramCountsAll.put(gram,oldCount+1);
        } else {
          ngramCountsAll.put(gram,1);
        }
        ++n;
      } // for (fin)

    } // for (st)

    return ngramCountsAll;

  }

  enum EffectiveLengthMethod {
    CLOSEST,
    SHORTEST,
    AVERAGE
  }

}
