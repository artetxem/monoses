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

import java.math.*;
import java.util.*;
import java.io.*;

public class BLEU_thresholdedTER extends TERMinusBLEU
{
  // This is a BLEU score, "conditioned" on a TER threshold

  // NOTE: even though this class extends TERMinusBLEU, it is *not* a TER-BLEU scoring metric.
  //       The score is simply the BLEU score, unless it doesn't meet some TER threshold, in
  //       which case it is a negated version of the BLEU score.  The only reason this class
  //       extends TERMinusBLEU is that (like TERMinusBLEU) it needs a BLEU scorer and
  //       a TER scorer.

  // individual components
  private double TER_threshold;

  public BLEU_thresholdedTER() { super(); }

  public BLEU_thresholdedTER(String[] Metric_options)
  {
    // M_o[0]: case sensitivity, case/nocase
    // M_o[1]: with-punctuation, punc/nopunc
    // M_o[2]: beam width, positive integer
    // M_o[3]: maximum shift distance, positive integer
    // M_o[4]: filename of tercom jar file
    // M_o[5]: number of threads to use for TER scoring (= number of tercom processes launched)
    // M_o[6]: maximum gram length, positive integer
    // M_o[7]: effective length calculation method, closest/shortest/average
    // M_o[8]: TER threshold

    // for 0-3, default values in tercom-0.7.25 are: nocase, punc, 20, 50

    myTER = new TER(Metric_options);
    myBLEU = new BLEU(Integer.parseInt(Metric_options[6]),Metric_options[7]);
    TER_threshold = Double.parseDouble(Metric_options[8]);

    initialize(); // set the data members of the metric
  }

  protected void initialize()
  {
    metricName = "BLEU_TER-th";
    toBeMinimized = false;
    suffStatsCount_TER = myTER.get_suffStatsCount();
    suffStatsCount_BLEU = myBLEU.get_suffStatsCount();
    suffStatsCount = suffStatsCount_TER + suffStatsCount_BLEU;
  }

  public double bestPossibleScore() { return 1.0; }
  public double worstPossibleScore() { return -1.0; }
    // -1.0 is a BLEU score of 0.0 that does not meet the TER threshold
    // In general, a score of -a means a BLEU score of 1-a that does
    // not meet the TER threshold (e.g. -0.60 means a BLEU score of 0.40
    //                             that does not meet the TER threshold)
    // i.e., BLEU scores that do not meet the threshold are made negative
    // to eliminate them from "competition" for the best score.  However
    // we don't map all BLEU scores to a single negative value, because
    // we'd still want to know which ones are better than others (within
    // the scores that do not meet the TER threshold)
    // This way, every BLEU score that does not meet the threshold (even
    // if it is a very high BLEU score) will be considered worse than
    // a BLEU score that does meet the threshold (even if it is very low
    // BLEU score).

  public double score(int[] stats)
  {
    if (stats.length != suffStatsCount) {
      System.out.println("Mismatch between stats.length and suffStatsCount (" + stats.length + " vs. " + suffStatsCount + ") in BLEU_thresholdedTER.score(int[])");
      System.exit(1);
    }

    double sc = 0.0;

    int[] stats_TER = new int[suffStatsCount_TER];
    int[] stats_BLEU = new int[suffStatsCount_BLEU];
    for (int s = 0; s < suffStatsCount_TER; ++s) { stats_TER[s] = stats[s]; }
    for (int s = 0; s < suffStatsCount_BLEU; ++s) { stats_BLEU[s] = stats[s+suffStatsCount_TER]; }

    double sc_T = myTER.score(stats_TER);
    double sc_B = myBLEU.score(stats_BLEU);

// the only place where BLEU_TER-th differs from TER-BLEU /* ~~~ */
/* ~~~ */
//    sc = sc_T - sc_B;
/* ~~~ */

/* ~~~ */
    if (sc_T > TER_threshold) { // TER score not good enough; return a "negated" version of the BLEU score
      sc = sc_B - 1.0;
        // why -1.0?  Just in case none of the scores meets the threshold, then at
        // least we back off to the best BLEU
    } else { // TER threshold is satisfied; return regular BLEU score
      sc = sc_B;
    }
/* ~~~ */

    return sc;
  }

}

