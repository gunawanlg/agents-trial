select
1/(1+exp(-s.LINEAR_SCORE)) as SCORE,
s.*
from (
    select
    w.feature_a_WOE * -0.6373764603049858
     + w.featureB_WOE * -0.5952350999512949
     + w.featC_WOE * -0.8643582973975634
     + w.feat_pred_D_WOE * -0.7098298562716113
     + w.featE_VAL * 0.4582281228917229
     + w.featF_v3_0_VAL * 0.576921144130071
     + w.featG_V2_VAL * 0.5623909066531244
     + w.featH_v4_0_VAL * 0.8358362941961892
     + w.feati_v3_VAL * 0.6175631752093271
     + w.var_v2_VAL * 0.719355781611567
     + w.Intercept * 7.864808103029582
    as LINEAR_SCORE,
    w.*
    from (
        select
        case
            when indosat_v2 < 0.030322997830808163 then 0.8589883180461642
            when indosat_v2 < 0.05057848058640957 then 0.14506493043860003
            when indosat_v2 >= 0.05057848058640957 then -0.4838054685680173
            when indosat_v2 is null then -0.008113478679658392
            else -0.008113478679658392
        end as feature_a_WOE,
        case
            when featureB = 'No Score' then -0.19802888585164036
            when featureB = 'No Match' then -0.19802888585164036
            when featureB < 556.5 then -0.8755047074479125
            when featureB < 582.5 then -0.2835526717920107
            when featureB < 637.5 then 0.3509134596093779
            when featureB >= 637.5 then 1.2771948599668872
            when featureB is null then -0.19802888585164036
            else -0.19802888585164036
        end as featureB_WOE,
        case
            when featC < 0.1297735944390297 then -0.31983068733407105
            when featC >= 0.1297735944390297 then -0.9408524915552476
            when featC is null then 0.04965688662575474
            else 0.04965688662575474
        end as featC_WOE,
        case
            when feat_pred_D = '0-1month' then -0.37442094147664373
            when feat_pred_D = '1-2month' then -0.37442094147664373
            when feat_pred_D = '10-11month' then -0.37442094147664373
            when feat_pred_D = '11-12month' then -0.37442094147664373
            when feat_pred_D = '12-13month' then -0.37442094147664373
            when feat_pred_D = '13-14month' then -0.37442094147664373
            when feat_pred_D = '14-15month' then -0.37442094147664373
            when feat_pred_D = '15-16month' then -0.37442094147664373
            when feat_pred_D = '16-17month' then -0.37442094147664373
            when feat_pred_D = '17-18month' then -0.37442094147664373
            when feat_pred_D = '18-19month' then -0.37442094147664373
            when feat_pred_D = '19-20month' then -0.37442094147664373
            when feat_pred_D = '2-3month' then -0.37442094147664373
            when feat_pred_D = '20-21month' then -0.37442094147664373
            when feat_pred_D = '21-22month' then -0.37442094147664373
            when feat_pred_D = '22-23month' then -0.37442094147664373
            when feat_pred_D = '23-24month' then -0.37442094147664373
            when feat_pred_D = '24-25month' then -0.37442094147664373
            when feat_pred_D = '25-26month' then -0.37442094147664373
            when feat_pred_D = '26-27month' then -0.37442094147664373
            when feat_pred_D = '27-28month' then -0.37442094147664373
            when feat_pred_D = '28-29month' then -0.37442094147664373
            when feat_pred_D = '29-30month' then -0.37442094147664373
            when feat_pred_D = '3-4month' then -0.37442094147664373
            when feat_pred_D = '30-31month' then -0.37442094147664373
            when feat_pred_D = '31-32month' then -0.37442094147664373
            when feat_pred_D = '32-33month' then -0.37442094147664373
            when feat_pred_D = '33-34month' then -0.13021532920078416
            when feat_pred_D = '34-35month' then -0.13021532920078416
            when feat_pred_D = '35-36month' then -0.13021532920078416
            when feat_pred_D = '36-37month' then -0.13021532920078416
            when feat_pred_D = '37-38month' then -0.13021532920078416
            when feat_pred_D = '38-39month' then -0.13021532920078416
            when feat_pred_D = '39-40month' then -0.13021532920078416
            when feat_pred_D = '4-5month' then -0.37442094147664373
            when feat_pred_D = '40-41month' then -0.13021532920078416
            when feat_pred_D = '41-42month' then -0.13021532920078416
            when feat_pred_D = '42-43month' then -0.13021532920078416
            when feat_pred_D = '43-44month' then -0.13021532920078416
            when feat_pred_D = '44-45month' then -0.13021532920078416
            when feat_pred_D = '45-46month' then -0.13021532920078416
            when feat_pred_D = '46-47month' then -0.13021532920078416
            when feat_pred_D = '47-48month' then -0.13021532920078416
            when feat_pred_D = '48-49month' then -0.13021532920078416
            when feat_pred_D = '49-50month' then -0.13021532920078416
            when feat_pred_D = '5-6month' then -0.37442094147664373
            when feat_pred_D = '50-51month' then 0.09577533753541978
            when feat_pred_D = '51-52month' then 0.09577533753541978
            when feat_pred_D = '52-53month' then 0.09577533753541978
            when feat_pred_D = '53-54month' then 0.09577533753541978
            when feat_pred_D = '54-55month' then 0.09577533753541978
            when feat_pred_D = '55-56month' then 0.09577533753541978
            when feat_pred_D = '56-57month' then 0.09577533753541978
            when feat_pred_D = '57-58month' then 0.09577533753541978
            when feat_pred_D = '58-59month' then 3.6602873860102516
            when feat_pred_D = '59-60month' then 3.6602873860102516
            when feat_pred_D = '6-7month' then -0.37442094147664373
            when feat_pred_D = '60-61month' then 3.6602873860102516
            when feat_pred_D = '61-62month' then 3.6602873860102516
            when feat_pred_D = '62-63month' then 3.6602873860102516
            when feat_pred_D = '63-64month' then 3.6602873860102516
            when feat_pred_D = '64-65month' then 3.6602873860102516
            when feat_pred_D = '65-66month' then 3.6602873860102516
            when feat_pred_D = '66-67month' then 3.6602873860102516
            when feat_pred_D = '67-68month' then 3.6602873860102516
            when feat_pred_D = '68-69month' then 3.6602873860102516
            when feat_pred_D = '7-8month' then -0.37442094147664373
            when feat_pred_D = '8-9month' then -0.37442094147664373
            when feat_pred_D = '9-10month' then -0.37442094147664373
            when feat_pred_D is null then 0.09577533753541978
            else 3.6602873860102516
        end as feat_pred_D_WOE,
        case
            when 1=1 then nvl(LN(featE/(1-featE)),-2.701124677318522)
            else 0
        end as featE_VAL,
        case
            when 1=1 then LN(featF_v3_0/(1-featF_v3_0))
            else 0
        end as featF_v3_0_VAL,
        case
            when 1=1 then nvl(LN(featG_V2/(1-featG_V2)),-2.97553316366986)
            else 0
        end as featG_V2_VAL,
        case
            when 1=1 then nvl(LN(featH_v4_0/(1-featH_v4_0)),-2.730217)
            else 0
        end as featH_v4_0_VAL,
        case
            when 1=1 then nvl(LN(feati_v3/(1-feati_v3)),-2.557910899788164)
            else 0
        end as feati_v3_VAL,
        case
            when 1=1 then nvl(LN(var_v2/(1-var_v2)),-2.7356004290961025)
            else 0
        end as var_v2_VAL,
        case
            when 1=1 then 1.0
            else 0
        end as Intercept
        from _SOURCETABLENAME_
    ) w
) s
