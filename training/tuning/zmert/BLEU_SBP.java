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

public class BLEU_SBP extends BLEU
{
  // constructors
  public BLEU_SBP() { super(); }
  public BLEU_SBP(String[] BLEU_SBP_options) { super(BLEU_SBP_options); }
  public BLEU_SBP(int mxGrmLn,String methodStr) { super(mxGrmLn,methodStr); }



  public int[] suffStats(String cand_str, int i)
  {
    int[] stats = new int[suffStatsCount];
    stats[0] = 1;

    String[] words = cand_str.split("\\s+");

//int wordCount = words.length;
//for (int j = 0; j < wordCount; ++j) { words[j] = words[j].intern(); }

    set_prec_suffStats(stats,words,i);

// the only place where BLEU_SBP differs from BLEU /* ~~~ */
/* ~~~ */
//    stats[maxGramLength+1] = words.length;
//    stats[maxGramLength+2] = effLength(words.length,i);
/* ~~~ */

/* ~~~ */
    int effectiveLength = effLength(words.length,i);
    stats[maxGramLength+1] = Math.min(words.length,effectiveLength);
    stats[maxGramLength+2] = effectiveLength;
/* ~~~ */

    return stats;
  }

}
