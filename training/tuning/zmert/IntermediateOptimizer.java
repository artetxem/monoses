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
import java.io.*;
import java.text.DecimalFormat;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;

public class IntermediateOptimizer implements Runnable
{
  /* non-static data members */
  private int j;
  private Semaphore blocker;
  private Vector<String> threadOutput;
  private String strToPrint;

  private double[] initialLambda;
  private double[] finalLambda;
  private int[][] best1Cand_suffStats;
  private double[] finalScore;
  private int[] candCount;
  private double[][][] featVal_array;
  private ConcurrentHashMap<Integer,int[]>[] suffStats_array;

  /* static data members */
  private final static DecimalFormat f4 = new DecimalFormat("###0.0000");
  private final static double NegInf = (-1.0 / 0.0);
  private final static double PosInf = (+1.0 / 0.0);

  private static int numSentences;
  private static int numDocuments;
  private static int[] docOfSentence;
  private static int docSubset_firstRank;
  private static int docSubset_lastRank;
  private static boolean optimizeSubset;
  private static int numParams;
  private static double[] normalizationOptions;
  private static boolean[] isOptimizable;
  private static double[] minThValue;
  private static double[] maxThValue;
  private static boolean oneModificationPerIteration;
  private static EvaluationMetric evalMetric;
  private static String metricName;
  private static String metricName_display;
  private static int suffStatsCount;
  private static String tmpDirPrefix;
  private static int verbosity;

  public static void set_MERTparams(
      int in_numSentences, int in_numDocuments, int[] in_docOfSentence, int[] in_docSubsetInfo,
      int in_numParams, double[] in_normalizationOptions,
      boolean[] in_isOptimizable, double[] in_minThValue, double[] in_maxThValue,
      boolean in_oneModificationPerIteration, EvaluationMetric in_evalMetric,
      String in_tmpDirPrefix, int in_verbosity)
  {
    numSentences = in_numSentences;
    numDocuments = in_numDocuments;
    docOfSentence = in_docOfSentence;

    docSubset_firstRank = in_docSubsetInfo[1];
    docSubset_lastRank = in_docSubsetInfo[2];
    if (in_docSubsetInfo[3] != numDocuments) optimizeSubset = true;
    else optimizeSubset = false;

    numParams = in_numParams;
    normalizationOptions = in_normalizationOptions;
    isOptimizable = in_isOptimizable;
    minThValue = in_minThValue;
    maxThValue = in_maxThValue;
    oneModificationPerIteration = in_oneModificationPerIteration;
    evalMetric = in_evalMetric;
    metricName = evalMetric.get_metricName();
    metricName_display = metricName;
    if (numDocuments > 1) metricName_display = "doc-level " + metricName;
    suffStatsCount = evalMetric.get_suffStatsCount();
    tmpDirPrefix = in_tmpDirPrefix;
    verbosity = in_verbosity;
  }

  public IntermediateOptimizer(
      int in_j, Semaphore in_blocker, Vector<String> in_threadOutput,
      double[] in_initialLambda, double[] in_finalLambda, int[][] in_best1Cand_suffStats,
      double[] in_finalScore, int[] in_candCount, double[][][] in_featVal_array,
      ConcurrentHashMap<Integer,int[]>[] in_suffStats_array)
  {
    j = in_j;
    blocker = in_blocker;
    threadOutput = in_threadOutput;
    strToPrint = "";

    initialLambda = in_initialLambda;
    finalLambda = in_finalLambda;
    best1Cand_suffStats = in_best1Cand_suffStats;
    finalScore = in_finalScore;
    candCount = in_candCount;
    featVal_array = in_featVal_array;
    suffStats_array = in_suffStats_array;
  }

//  private TreeMap<Double,TreeMap> thresholdsForParam(int c, int[] candCount, double[][][] featVal_array, double[] currLambda, TreeSet<Integer>[] indicesOfInterest)
  private void set_thresholdsForParam(
      TreeMap<Double,TreeMap<Integer,int[]>> thresholdsAll, int c,
      double[] currLambda, TreeSet<Integer>[] indicesOfInterest)
  {
/*
    TreeMap[] thresholds = new TreeMap[numSentences];
      // thresholds[i] stores thresholds for the cth parameter obtained by
      // processing the candidates of sentence i.  It not only stores the
      // thresholds themselves, but also a triple of {i,from,to}, where from/to
      // are indices that characterize the 1-best switch at this threshold.

    for (int i = 0; i < numSentences; ++i) {
      thresholds[i] = new TreeMap<Double,int[]>();
    }
*/

//    TreeMap<Double,int[]> thresholds = new TreeMap<Double,int[]>();

    // Find threshold points
//    TreeMap<Double,TreeMap> thresholdsAll = new TreeMap<Double,TreeMap>();
    thresholdsAll.clear();

    int ipCount = 0;
    for (int i = 0; i < numSentences; ++i) {
    // find threshold points contributed by ith sentence

//      println("Processing sentence #" + i,4);

      int numCandidates = candCount[i];
        // aka simply K

      double[] slope = new double[numCandidates];
        // will be h_c from candidatesInfo
        // repeated here for easy access
      double[] offset = new double[numCandidates];
        // SUM_j!=c currLambda_j*h_j(x)

      int minSlopeIndex = -1;          // index of line with steepest descent...
      double minSlope = PosInf;        // ...and its slope...
      double offset_minSlope = NegInf; // ...and its offset (needed to break ties)

      int maxSlopeIndex = -1;          // index of line with steepest ascent...
      double maxSlope = NegInf;        // ...and its slope...
      double offset_maxSlope = NegInf; // ...and its offset (needed to break ties)

      double bestScore_left = NegInf;  // these are used if the min/max values are
      double bestScore_right = NegInf; // not neg/pos infinity

      for (int k = 0; k < numCandidates; ++k) {
        slope[k] = featVal_array[c][i][k];

        offset[k] = 0.0;
        for (int c2 = 1; c2 <= numParams; ++c2) {
          if (c2 != c) { offset[k] += currLambda[c2]*featVal_array[c2][i][k]; }
        }

        // debugging
//        println("@ (i,k)=(" + i + "," + k + "), "
//               + "slope = " + slope[k] + "; offset = " + offset[k],4);

        if (minThValue[c] == NegInf) {
          if (slope[k] < minSlope || (slope[k] == minSlope && offset[k] > offset_minSlope)) {
            minSlopeIndex = k;
            minSlope = slope[k];
            offset_minSlope = offset[k];
          }
        } else {
          double score = offset[k] + ((minThValue[c]-0.1)*slope[k]);
          if (score > bestScore_left || (score == bestScore_left && slope[k] > minSlope)) {
            minSlopeIndex = k;
            minSlope = slope[k];
            bestScore_left = score;
          }
        }

        if (maxThValue[c] == PosInf) {
          if (slope[k] > maxSlope || (slope[k] == maxSlope && offset[k] > offset_maxSlope)) {
            maxSlopeIndex = k;
            maxSlope = slope[k];
            offset_maxSlope = offset[k];
          }
        } else {
          double score = offset[k] + ((maxThValue[c]+0.1)*slope[k]);
          if (score > bestScore_right || (score == bestScore_right && slope[k] < maxSlope)) {
            maxSlopeIndex = k;
            maxSlope = slope[k];
            bestScore_right = score;
          }
        }
      }

      // debugging
//      println("minSlope is @ k = " + minSlopeIndex + ": slope " + minSlope
//            + " (offset " + offset_minSlope + ")",4);
//      println("maxSlope is @ k = " + maxSlopeIndex + ": slope " + maxSlope
//            + " (offset " + offset_maxSlope + ")",4);


      // some lines can be eliminated: the ones that have a lower offset
      // than some other line with the same slope.
      // That is, for any k1 and k2:
      //   if slope[k1] = slope[k2] and offset[k1] > offset[k2],
      //   then k2 can be eliminated.
      // (This is actually important to do as it eliminates a bug.)
//      HashSet<Integer> discardedIndices = indicesToDiscard(slope,offset);


//      println("Extracting thresholds[(i,c)=(" + i + "," + c + ")]",4);

      int currIndex = minSlopeIndex;
        // As we traverse the currLambda_c dimension, the "winner" candidate will
        // change at intersection points.  currIndex tells us which candidate is
        // the winner in the interval currently under investigation.

        // We traverse the lambda_c dimension starting at -Inf.  The line with
        // steepest descent is the winner as lambda_c -> -Inf, so we initialize
        // currIndex to minSlopeIndex to reflect that fact.

        // Similarly, the winner as lambda_c -> +Inf is the line with the
        // steepest *ascent* (i.e. max slope), and so we continue finding
        // intersection points until we hit that line.

        // Notice that we didn't have to investigate the entire space (-Inf,+Inf)
        // if the parameter's range is more restricted than that.  That is why, in
        // the loop above, the "left-most" winner is not necessarily the one with
        // the steepest descent (though it will be if minThValue[c] is -Inf).
        // And similarly, the "right-most" winner is not necessarily the one with
        // the steepest ascent (though it will be if minThValue[c] is +Inf).  The
        // point of doing this is to avoid extracting thresholds that will end up
        // being discarded anyway due to range constraints, thus saving us a little
        // bit of time.

      int last_new_k = -1;

      while (currIndex != maxSlopeIndex) {

        if (currIndex < 0) break;
          // Due to rounding errors, the index identified as maxSlopeIndex above
          // might be different from the one this loop expects, in which case
          // it won't be found and currIndex remains -1.  So if currIndex is -1
          // a rounding error happened, which is cool since we can just break.

//        print("cI=" + currIndex + " ",4);

        // find the candidate whose line is the first to intersect the current
        // line.  ("first" meaning with an intersection point that has the
        //         lowest possible lambda_c value.)

        double nearestIntersectionPoint = PosInf;
        int nearestIntersectingLineIndex = -1;

        for (int k = 0; k < numCandidates; ++k) {
//          if (slope[k] > slope[currIndex] && !discardedIndices.contains(k)) {
          if (slope[k] > slope[currIndex]) {
          // only higher-sloped lines will intersect the current line
          // (If we didn't have discardedIndices a bug would creep up here.)

            // find intersection point ip_k
            double ip_k = (offset[k] - offset[currIndex])/(slope[currIndex] - slope[k]);
            if (ip_k < nearestIntersectionPoint) {
              nearestIntersectionPoint = ip_k;
              nearestIntersectingLineIndex = k;
            }
          }
        }

//        print("ip=" + f4.format(nearestIntersectionPoint) + " ",4);
        ++ipCount;

        if (nearestIntersectionPoint > minThValue[c] && nearestIntersectionPoint < maxThValue[c]) {

          int[] th_info = {currIndex,nearestIntersectingLineIndex};
          last_new_k = nearestIntersectingLineIndex;

          indicesOfInterest[i].add(currIndex); // old_k
//          indicesOfInterest_all[i].add(currIndex); // old_k   ***/

          if (!thresholdsAll.containsKey(nearestIntersectionPoint)) {
            TreeMap<Integer,int[]> A = new TreeMap<Integer,int[]>();
            A.put(i,th_info);
            thresholdsAll.put(nearestIntersectionPoint,A);
          } else {
            TreeMap<Integer,int[]> A = thresholdsAll.get(nearestIntersectionPoint);
            if (!A.containsKey(i)) {
              A.put(i,th_info);
            } else {
              int[] old_th_info = A.get(i);
              old_th_info[1] = th_info[1]; // replace the existing new_k
              A.put(i,th_info);
            }
            thresholdsAll.put(nearestIntersectionPoint,A);
          }
/*
          if (!thresholds.containsKey(nearestIntersectionPoint)) {
            thresholds.put(nearestIntersectionPoint,th_info);
              // i.e., at lambda_c = nIP, the (index of the) 1-best changes
              // from currIndex to nearestIntersectingLineIndex (which is
              // indicated in th_info)
          } else { // extremely rare, but causes problem if it does occur
            // in essence, just replace the new_k of the existing th_info
            int[] old_th_info = (int[])thresholds.get(nearestIntersectionPoint);
            old_th_info[1] = th_info[1];
            thresholds.put(nearestIntersectionPoint,old_th_info);
            // When does this happen?  If two consecutive intersection points are so close
            // to each other so as to appear as having the same value.  For instance, assume
            // we have two intersection points ip1 and ip2 corresponding to two transitions,
            // one from k_a to k_b, and the other from k_b to k_c.  It might be the case
            // that ip2-ip1 is extremeley small, so that the ip2 entry would actually REPLACE
            // the ip1 entry.  This would be bad.

            // Instead, we pretend that k_b never happened, and just assume there is a single
            // intersection point, ip (which equals whatever value Java calculates for ip1
            // and ip2), with a corresponding transition of k_a to k_c.
          }
*/
          } // if (in-range)

        currIndex = nearestIntersectingLineIndex;

      } // end while (currIndex != maxSlopeIndex)

      if (last_new_k != -1) {
        indicesOfInterest[i].add(last_new_k); // last new_k
//        indicesOfInterest_all[i].add(last_new_k); // last new_k  ***/
      }

//      println("cI=" + currIndex + "(=? " + maxSlopeIndex + " = mxSI)",4);

      // now thresholds has the values for lambda_c at which score changes
      // based on the candidates for the ith sentence

//      println("",4);

/*
      Iterator<Double> It = (thresholds.keySet()).iterator();
      int[] th_info = null;
      while (It.hasNext()) { // process intersection points contributed by this sentence
        double ip = It.next();
        if (ip > minThValue[c] && ip < maxThValue[c]) {
          th_info = thresholds.get(ip);
          if (!thresholdsAll.containsKey(ip)) {
            TreeMap A = new TreeMap();
            A.put(i,th_info);
            thresholdsAll.put(ip,A);
          } else {
            // not frequent, but does happen (when same intersection point
            // corresponds to a candidate switch for more than one i)
            TreeMap A = thresholdsAll.get(ip);
            A.put(i,th_info);
            thresholdsAll.put(ip,A);
          }

//          if (useDisk == 2) {
            // th_info[0] = old_k, th_info[1] = new_k
            indicesOfInterest[i].add(th_info[0]);
//          }

        } // if (in-range)

      } // while (It.hasNext())
*/

/*
//      if (useDisk == 2 && th_info != null) {
      if (th_info != null) {
        // new_k from the last th_info (previous new_k already appear as the next old_k)
        indicesOfInterest[i].add(th_info[1]);
      }
*/

//      thresholds.clear();

    } // for (i)

    // now thresholdsAll has the values for lambda_c at which score changes
    // based on the candidates for *all* the sentences (that satisfy
    // range constraints).
    // Each lambda_c value maps to a Vector of th_info.  An overwhelming majority
    // of these Vectors are of size 1.

    // indicesOfInterest[i] tells us which candidates for the ith sentence need
    // to be read from the merged decoder output file.

    if (thresholdsAll.size() != 0) {
      double smallest_th = thresholdsAll.firstKey();
      double largest_th = thresholdsAll.lastKey();
      println("# extracted thresholds: " + thresholdsAll.size(),2);
      println("Smallest extracted threshold: " + smallest_th,2);
      println("Largest extracted threshold: " + largest_th,2);

      if (maxThValue[c] != PosInf) {
        thresholdsAll.put(maxThValue[c],null);
      } else {
        thresholdsAll.put((thresholdsAll.lastKey() + 0.1),null);
      }
    }

//    return thresholdsAll;

  } // TreeMap<Double,TreeMap> thresholdsForParam (int c)

  private double[] line_opt(
      TreeMap<Double,TreeMap<Integer,int[]>> thresholdsAll, int[] indexOfCurrBest,
      int c, double[] lambda)
  {
    println("Line-optimizing lambda[" + c + "]...",3);

    double[] bestScoreInfo = new double[2];
      // to be returned: [0] will store the best lambda, and [1] will store its score

    if (thresholdsAll.size() == 0) {
      // no thresholds extracted!  Possible in theory...
      // simply return current value for this parameter
      println("No thresholds extracted!  Returning this parameter's current value...",2);

      bestScoreInfo[0] = lambda[c];
      bestScoreInfo[1] = evalMetric.worstPossibleScore();

      return bestScoreInfo;
    }

    double smallest_th = thresholdsAll.firstKey();
    double largest_th = thresholdsAll.lastKey();
    println("Minimum threshold: " + smallest_th,3);
    println("Maximum threshold: " + largest_th,3);

    double[] temp_lambda = new double[1+numParams];
    System.arraycopy(lambda,1,temp_lambda,1,numParams);

    double ip_prev = 0.0, ip_curr = 0.0;

    if (minThValue[c] != NegInf) {
      temp_lambda[c] = (minThValue[c] + smallest_th) / 2.0;
      ip_curr = minThValue[c];
    } else {
      temp_lambda[c] = smallest_th - 0.05;
      ip_curr = smallest_th - 0.1;
    }




    int[][] suffStats = new int[numSentences][suffStatsCount];
      // suffStats[i][s] stores the contribution to the sth sufficient
      // statistic from the candidate for the ith sentence (the candidate
      // indicated by indexOfCurrBest[i]).

    int[][] suffStats_doc = new int[numDocuments][suffStatsCount];
      // suffStats_doc[doc][s] := SUM_i suffStats[i][s], over sentences in the doc'th document
      // i.e. treat each document as a mini corpus
      // (if not doing document-level optimization, all sentences will belong in a single
      //  document: the 1st one, indexed 0)

    // initialize document SS
    for (int doc = 0; doc < numDocuments; ++doc) {
      for (int s = 0; s < suffStatsCount; ++s) {
        suffStats_doc[doc][s] = 0;
      }
    }

    // Now, set suffStats[][], and increment suffStats_doc[][]
    for (int i = 0; i < numSentences; ++i) {
      suffStats[i] = suffStats_array[i].get(indexOfCurrBest[i]);

      for (int s = 0; s < suffStatsCount; ++s) {
        suffStats_doc[docOfSentence[i]][s] += suffStats[i][s];
      }
    }



    double bestScore = 0.0;
    if (optimizeSubset) bestScore = evalMetric.score(suffStats_doc,docSubset_firstRank,docSubset_lastRank);
    else bestScore = evalMetric.score(suffStats_doc);
    double bestLambdaVal = temp_lambda[c];
    double nextLambdaVal = bestLambdaVal;
    println("At lambda[" + c + "] = " + bestLambdaVal + ","
          + "\t" + metricName_display + " = " + bestScore + " (*)",3);

    Iterator<Double> It = (thresholdsAll.keySet()).iterator();
    if (It.hasNext()) { ip_curr = It.next(); }

    while (It.hasNext()) {
      ip_prev = ip_curr;
      ip_curr = It.next();
      nextLambdaVal = (ip_prev + ip_curr)/2.0;

      TreeMap<Integer,int[]> th_info_M = thresholdsAll.get(ip_prev);
      Iterator<Integer> It2 = (th_info_M.keySet()).iterator();
      while (It2.hasNext()) {
        int i = It2.next();
          // i.e. the 1-best for the i'th sentence changes at this threshold value
        int docOf_i = docOfSentence[i];

        int[] th_info = th_info_M.get(i);
        @SuppressWarnings("unused")
        int old_k = th_info[0]; // should be equal to indexOfCurrBest[i]
        int new_k = th_info[1];

        for (int s = 0; s < suffStatsCount; ++s) {
          suffStats_doc[docOf_i][s] -= suffStats[i][s]; // subtract stats for candidate old_k
        }

        indexOfCurrBest[i] = new_k;
        suffStats[i] = suffStats_array[i].get(indexOfCurrBest[i]); // update the SS for the i'th sentence

        for (int s = 0; s < suffStatsCount; ++s) {
          suffStats_doc[docOf_i][s] += suffStats[i][s]; // add stats for candidate new_k
        }

      }

      double nextTestScore = 0.0;
      if (optimizeSubset) nextTestScore = evalMetric.score(suffStats_doc,docSubset_firstRank,docSubset_lastRank);
      else nextTestScore = evalMetric.score(suffStats_doc);

      print("At lambda[" + c + "] = " + nextLambdaVal + ","
          + "\t" + metricName_display + " = " + nextTestScore,3);

      if (evalMetric.isBetter(nextTestScore,bestScore)) {
        bestScore = nextTestScore;
        bestLambdaVal = nextLambdaVal;
        print(" (*)",3);
      }

      println("",3);

    } // while (It.hasNext())

    println("",3);

    // what is the purpose of this block of code ?????????????????????
/*
    if (maxThValue[c] != PosInf) {
      nextLambdaVal = (largest_th + maxThValue[c]) / 2.0;
    } else {
      nextLambdaVal = largest_th + 0.05;
    }
*/
    // ???????????????????????????????????????????????????????????????

    /*************************************************/
    /*************************************************/

    bestScoreInfo[0] = bestLambdaVal;
    bestScoreInfo[1] = bestScore;

    return bestScoreInfo;

  } // double[] line_opt(int c)

  private void set_suffStats_array(TreeSet<Integer>[] indicesOfInterest)
  {
    int candsOfInterestCount = 0;
    int candsOfInterestCount_all = 0;
    for (int i = 0; i < numSentences; ++i) {
      candsOfInterestCount += indicesOfInterest[i].size();
//      candsOfInterestCount_all += indicesOfInterest_all[i].size();  ****/
    }
    println("Processing merged stats file; extracting SS "
          + "for " + candsOfInterestCount + " candidates of interest.",2);
//    println("(*_all: " + candsOfInterestCount_all + ")",2); *****/


    try {

      // process the merged sufficient statistics file, and read (and store) the
      // stats for candidates of interest
      BufferedReader inFile = new BufferedReader(new FileReader(tmpDirPrefix+"temp.stats.merged"));
      String candidate_suffStats;

      for (int i = 0; i < numSentences; ++i) {
        int numCandidates = candCount[i];

        int currCand = 0;
        Iterator<Integer> It = indicesOfInterest[i].iterator();

        while (It.hasNext()) {
          int nextIndex = It.next();

          // skip candidates until you get to the nextIndex'th candidate
          while (currCand < nextIndex) {
            inFile.readLine();
            ++currCand;
          }

          // now currCand == nextIndex, and the next line in inFile
          // contains the sufficient statistics we want

          candidate_suffStats = inFile.readLine();
          ++currCand;

          String[] suffStats_str = candidate_suffStats.split("\\s+");

          int[] suffStats = new int[suffStatsCount];

          for (int s = 0; s < suffStatsCount; ++s) {
            suffStats[s] = Integer.parseInt(suffStats_str[s]);
          }

          suffStats_array[i].put(nextIndex,suffStats);

        }

        // skip the rest of ith sentence's candidates
        while (currCand < numCandidates) {
          inFile.readLine();
          ++currCand;
        }

      } // for (i)

      inFile.close();

    } catch (FileNotFoundException e) {
      System.err.println("FileNotFoundException in MertCore.initialize(int): " + e.getMessage());
      System.exit(99901);
    } catch (IOException e) {
      System.err.println("IOException in MertCore.initialize(int): " + e.getMessage());
      System.exit(99902);
    }

  } // set_suffStats_array(HashMap[] suffStats_array, TreeSet[] indicesOfInterest, Vector[] candidates)

  private double L_norm(double[] A, double pow)
  {
    // calculates the L-pow norm of A[]
    // NOTE: this calculation ignores A[0]
    double sum = 0.0;
    for (int i = 1; i < A.length; ++i) {
      sum += Math.pow(Math.abs(A[i]),pow);
    }
    return Math.pow(sum,1/pow);
  }

  private int[] initial_indexOfCurrBest(double[] temp_lambda, TreeSet<Integer>[] indicesOfInterest)
  {
    int[] indexOfCurrBest = new int[numSentences];
      // As we traverse lambda_c, indexOfCurrBest indicates which is the
      // current best candidate.

    // initialize indexOfCurrBest[]

    for (int i = 0; i < numSentences; ++i) {
      int numCandidates = candCount[i];

      double max = NegInf;
      int indexOfMax = -1;
      for (int k = 0; k < numCandidates; ++k) {
        double score = 0;

        for (int c2 = 1; c2 <= numParams; ++c2) {
          score += temp_lambda[c2] * featVal_array[c2][i][k];
        }
        if (score > max) {
          max = score;
          indexOfMax = k;
        }
      }

      indexOfCurrBest[i] = indexOfMax;

//      if (useDisk == 2) {
        // add indexOfCurrBest[i] to indicesOfInterest
        indicesOfInterest[i].add(indexOfMax);
//        indicesOfInterest_all[i].add(indexOfMax);
//      }

    }

    return indexOfCurrBest;

  } // int[] initial_indexOfCurrBest (int c)

  private double[] bestParamToChange(TreeMap<Double,TreeMap<Integer,int[]>>[] thresholdsAll, int lastChanged_c, double[] currLambda)
  {
    int c_best = 0; // which parameter to change?
    double bestLambdaVal = 0.0;
    double bestScore;
    if (evalMetric.getToBeMinimized()) {
      bestScore = evalMetric.worstPossibleScore() + 1.0;
    } else {
      bestScore = evalMetric.worstPossibleScore() - 1.0;
    }




    // prep for line_opt

    TreeSet<Integer>[] indicesOfInterest = null;
    // indicesOfInterest[i] tells us which candidates for the ith sentence need
    // to be read from the merged decoder output file.

//    if (useDisk == 2) {
      @SuppressWarnings("unchecked")
      TreeSet<Integer>[] temp_TSA = new TreeSet[numSentences];
      indicesOfInterest = temp_TSA;
      for (int i = 0; i < numSentences; ++i) {
        indicesOfInterest[i] = new TreeSet<Integer>();
      }
//    }

    int[][] indexOfCurrBest = new int[1+numParams][numSentences];

    for (int c = 1; c <= numParams; ++c) {
      if (!isOptimizable[c]) {
        println("Not investigating lambda[j=" + j + "][" + c + "].",2);
      } else {
        if (c != lastChanged_c) {
          println("Investigating lambda[j=" + j + "][" + c + "]...",2);
//          thresholdsAll[c] = thresholdsForParam(c,candCount,featVal_array,currLambda,indicesOfInterest);
          set_thresholdsForParam(
            thresholdsAll[c],c,currLambda,indicesOfInterest);
        } else {
          println("Keeping thresholds for lambda[j=" + j + "][" + c + "] from previous step.",2);
        }
        // now thresholdsAll has the values for lambda_c at which score changes
        // based on the candidates for *all* the sentences (that satisfy
        // range constraints).
        // Each lambda_c value maps to a Vector of th_info.  An overwhelming majority
        // of these Vectors are of size 1.

        if (thresholdsAll[c].size() != 0) {

          double[] temp_lambda = new double[1+numParams];
          System.arraycopy(currLambda,1,temp_lambda,1,numParams);

          double smallest_th = thresholdsAll[c].firstKey();

          if (minThValue[c] != NegInf) {
            temp_lambda[c] = (minThValue[c] + smallest_th) / 2.0;
          } else {
            temp_lambda[c] = smallest_th - 0.05;
          }

          indexOfCurrBest[c] =
            initial_indexOfCurrBest(temp_lambda,indicesOfInterest);
        }
      }

      println("",2);

    }



//    if (useDisk == 2) {

      set_suffStats_array(indicesOfInterest);

//    } // if (useDisk == 2)



    for (int c = 1; c <= numParams; ++c) {
    // investigate currLambda[j][c]

      if (isOptimizable[c]) {
        double[] bestScoreInfo_c =
          line_opt(thresholdsAll[c],indexOfCurrBest[c],c,currLambda);
          // get best score and its lambda value

        double bestLambdaVal_c = bestScoreInfo_c[0];
        double bestScore_c = bestScoreInfo_c[1];

        if (evalMetric.isBetter(bestScore_c,bestScore)) {
          c_best = c;
          bestLambdaVal = bestLambdaVal_c;
          bestScore = bestScore_c;
        }

      } // if (!isOptimizable[c])

    }




    // delete according to indicesOfInterest

//    printMemoryUsage();

//    if (useDisk == 2) {

      for (int i = 0; i < numSentences; ++i) {

        indicesOfInterest[i].clear();

      }

//    }

//    cleanupMemory();
//    printMemoryUsage();
//    println("",2);








    double[] c_best_info = {c_best,bestLambdaVal,bestScore};
    return c_best_info;

  } // double[] bestParamToChange(int j, double[] currLambda)

  private void normalizeLambda(double[] origLambda)
  {
    // private String[] normalizationOptions;
      // How should a lambda[] vector be normalized (before decoding)?
      //   nO[0] = 0: no normalization
      //   nO[0] = 1: scale so that parameter nO[2] has absolute value nO[1]
      //   nO[0] = 2: scale so that the maximum absolute value is nO[1]
      //   nO[0] = 3: scale so that the minimum absolute value is nO[1]
      //   nO[0] = 4: scale so that the L-nO[1] norm equals nO[2]

    int normalizationMethod = (int)normalizationOptions[0];
    double scalingFactor = 1.0;
    if (normalizationMethod == 0) {

      scalingFactor = 1.0;

    } else if (normalizationMethod == 1) {

      int c = (int)normalizationOptions[2];
      scalingFactor = normalizationOptions[1]/Math.abs(origLambda[c]);

    } else if (normalizationMethod == 2) {

      double maxAbsVal = -1;
      int maxAbsVal_c = 0;
      for (int c = 1; c <= numParams; ++c) {
        if (Math.abs(origLambda[c]) > maxAbsVal) {
          maxAbsVal = Math.abs(origLambda[c]);
          maxAbsVal_c = c;
        }
      }
      scalingFactor = normalizationOptions[1]/Math.abs(origLambda[maxAbsVal_c]);

    } else if (normalizationMethod == 3) {

      double minAbsVal = PosInf;
      int minAbsVal_c = 0;
      for (int c = 1; c <= numParams; ++c) {
        if (Math.abs(origLambda[c]) < minAbsVal) {
          minAbsVal = Math.abs(origLambda[c]);
          minAbsVal_c = c;
        }
      }
      scalingFactor = normalizationOptions[1]/Math.abs(origLambda[minAbsVal_c]);

    } else if (normalizationMethod == 4) {

      double pow = normalizationOptions[1];
      double norm = L_norm(origLambda,pow);
      scalingFactor = normalizationOptions[2]/norm;

    }

    for (int c = 1; c <= numParams; ++c) {
      origLambda[c] *= scalingFactor;
    }

  }

  private void real_run() {
    @SuppressWarnings("unchecked")
    TreeMap<Double,TreeMap<Integer,int[]>>[] thresholdsAll = new TreeMap[1+numParams];
    thresholdsAll[0] = null;
    for (int c = 1; c <= numParams; ++c) {
      if (isOptimizable[c]) {
        thresholdsAll[c] = new TreeMap<Double,TreeMap<Integer,int[]>>();
      } else {
        thresholdsAll[c] = null;
      }
    }


//    cleanupMemory();

    println("+++ Optimization of lambda[j=" + j + "] starting @ " + (new Date()) + " +++",1);

    double[] currLambda = new double[1+numParams];
    System.arraycopy(initialLambda,1,currLambda,1,numParams);

    int[][] best1Cand_suffStats_doc = new int[numDocuments][suffStatsCount];
    for (int doc = 0; doc < numDocuments; ++doc) {
      for (int s = 0; s < suffStatsCount; ++s) {
        best1Cand_suffStats_doc[doc][s] = 0;
      }
    }

    for (int i = 0; i < numSentences; ++i) {
      for (int s = 0; s < suffStatsCount; ++s) {
        best1Cand_suffStats_doc[docOfSentence[i]][s] += best1Cand_suffStats[i][s];
      }
    }

    double initialScore = 0.0;
    if (optimizeSubset) initialScore = evalMetric.score(best1Cand_suffStats_doc,docSubset_firstRank,docSubset_lastRank);
    else initialScore = evalMetric.score(best1Cand_suffStats_doc);

    println("Initial lambda[j=" + j + "]: " + lambdaToString(initialLambda),1);
    println("(Initial " + metricName_display + "[j=" + j + "]: " + initialScore + ")",1);
    println("",1);
    finalScore[j] = initialScore;

    int c_best = 0; // which param to change?
    double bestLambdaVal = 0; // what value to change to?
    double bestScore = 0; // what score would be achieved?

    while (true) {

      double[] c_best_info = bestParamToChange(thresholdsAll,c_best,currLambda);
          // we pass in c_best because we don't need
          // to recalculate thresholds for it
      c_best = (int)c_best_info[0]; // which param to change?
      bestLambdaVal = c_best_info[1]; // what value to change to?
      bestScore = c_best_info[2]; // what score would be achieved?

      // now c_best is the parameter giving the most gain

      if (evalMetric.isBetter(bestScore,finalScore[j])) {
        println("*** Changing lambda[j=" + j + "][" + c_best + "] from "
              + f4.format(currLambda[c_best])
              + " (" + metricName_display + ": " + f4.format(finalScore[j]) + ") to "
              + f4.format(bestLambdaVal)
              + " (" + metricName_display + ": " + f4.format(bestScore) + ") ***",2);
        println("*** Old lambda[j=" + j + "]: " + lambdaToString(currLambda) + " ***",2);
        currLambda[c_best] = bestLambdaVal;
        finalScore[j] = bestScore;
        println("*** New lambda[j=" + j + "]: " + lambdaToString(currLambda) + " ***",2);
        println("",2);
      } else {
        println("*** Not changing any weight in lambda[j=" + j + "] ***",2);
        println("*** lambda[j=" + j + "]: " + lambdaToString(currLambda) + " ***",2);
        println("",2);
        break; // exit while (true) loop
      }

      if (oneModificationPerIteration) { break; } // exit while (true) loop

    } // while (true)

    // now currLambda is the optimized weight vector on the current candidate list
    // (corresponding to initialLambda)

    System.arraycopy(currLambda,1,finalLambda,1,numParams);
    normalizeLambda(finalLambda);
    // check if a lambda is outside its threshold range
    for (int c = 1; c <= numParams; ++c) {
      if (finalLambda[c] < minThValue[c] || finalLambda[c] > maxThValue[c]) {
        println("Warning: after normalization, final lambda[j=" + j + "][" + c + "]="
              + f4.format(finalLambda[c]) + " is outside its critical value range.",2);
      }
    }
    println("Final lambda[j=" + j + "]: " + lambdaToString(finalLambda),1);
    println("(Final " + metricName_display + "[j=" + j + "]: " + finalScore[j] + ")",1);
    println("",1);

    blocker.release();
  }

  public void run() {
    try {
      real_run();
    } catch (Exception e) {
      e.printStackTrace();
      System.err.println("Exception in IntermediateOptimizer.run(): " + e.getMessage());
      System.exit(99905);
    }
    if (!strToPrint.equals("")) {
      threadOutput.add(strToPrint);
    }
  }

  private void println(String str, int priority) { if (priority <= verbosity) println(str); }
  private void print(String str, int priority) { if (priority <= verbosity) print(str); }

  private void println(String str) { threadOutput.add(strToPrint + str); strToPrint = ""; }
  private void print(String str) { strToPrint += str; }

  private String lambdaToString(double[] lambdaA)
  {
    String retStr = "{";
    for (int c = 1; c <= numParams-1; ++c) {
      retStr += "" + lambdaA[c] + ", ";
    }
    retStr += "" + lambdaA[numParams] + "}";

    return retStr;
  }
}

