#' Cocaine data.
#'
#' @source https://github.com/CCS-Lab/easyml/blob/master/Python/datasets/cocaine_dependence.csv
#' @format Data frame with columns
#' \describe{
#' \item{subject}{Subject ID}
#' \item{diagnosis}{Diagnosis status}
#' \item{age}{Age of Subject}
#' \item{male}{Gender of subject}
#' \item{edu_yrs}{Education of subject in years}
#' \item{imt_comm_errors}{Immediate Memory Task commission errors}
#' \item{imt_omis_errors}{Immediate Memory Task omission errors}
#' \item{a_imt}{Immediate Memory Task non-parametric discriminability}
#' \item{b_d_imt}{Immediate Memory Task response bias }
#' \item{stop_ssrt}{Stop-Signal Task stop-signal reaction time}
#' \item{lnk_adjdd}{log(Adjusting Delay-Discounting Task discounting rate)}
#' \item{lkhat_kirby}{log(Kirby Monetary-Choice Delay-Discounting Questionnaire)}
#' \item{revlr_per_errors}{Probabilistic Reversal-Learning Task errors}
#' \item{bis_attention}{Barratt Impulsiveness Scale attentional impulsivity}
#' \item{bis_motor}{Barratt Impulsiveness Scale motor impulsivity}
#' \item{bis_nonpL}{Barratt Impulsiveness Scale nonplanning impulsivity}
#' \item{igt_total}{Iowa Gambling Task total score}
#' }
#' @family data
#' @examples
#' data("cocaine_dependence", package = "easyml")
"cocaine_dependence"

#' Prostate data.
#'
#' @source Stamey, T.A., Kabalin, J.N., McNeal, J.E., Johnstone, I.M., Freiha, F., Redwine, E.A. and Yang, N. (1989) Prostate specific antigen in the diagnosis and treatment of adenocarcinoma of the prostate: II. radical prostatectomy treated patients, Journal of Urology 141(5), 1076-1083. 
#' @format Data frame with columns
#' \describe{
#' \item{lcavol}{log(cancer volume)}
#' \item{lweight}{log(prostate weight)}
#' \item{age}{age}
#' \item{lbph}{log(benign prostatic hyperplasia amount)}
#' \item{svi}{seminal vesicle invasion}
#' \item{lcp}{log(capsular penetration)}
#' \item{gleason}{Gleason score}
#' \item{pgg45}{percentage Gleason scores 4 or 5}
#' \item{lpsa}{log(prostate specific antigen)}
#' }
#' @family data
#' @examples
#' data("prostate", package = "easyml")
"prostate"
