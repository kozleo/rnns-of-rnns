# Load in all of the neurogym tasks
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
from neurogym import Dataset
from Mod_Cog.mod_cog_tasks import *


def load_all_mod_cog_tasks():
    envs = [
    go(),
    rtgo(),
    dlygo(),
    anti(),
    rtanti(),
    dlyanti(),
    dm1(),
    dm2(),
    ctxdm1(),
    ctxdm2(),
    multidm(),
    dlydm1(),
    dlydm2(),
    ctxdlydm1(),
    ctxdlydm2(),
    multidlydm(),
    dms(),
    dnms(),
    dmc(),
    dnmc(),
    dlygointr(),
    dlygointl(),
    dlyantiintr(),
    dlyantiintl(),
    dlydm1intr(),
    dlydm1intl(),
    dlydm2intr(),
    dlydm2intl(),
    ctxdlydm1intr(),
    ctxdlydm1intl(),
    ctxdlydm2intr(),
    ctxdlydm2intl(),
    multidlydmintr(),
    multidlydmintl(),
    dmsintr(),
    dmsintl(),
    dnmsintr(),
    dnmsintl(),
    dmcintr(),
    dmcintl(),
    dnmcintr(),
    dnmcintl(),
    goseqr(),
    rtgoseqr(),
    dlygoseqr(),
    antiseqr(),
    rtantiseqr(),
    dlyantiseqr(),
    dm1seqr(),
    dm2seqr(),
    ctxdm1seqr(),
    ctxdm2seqr(),
    multidmseqr(),
    dlydm1seqr(),
    dlydm2seqr(),
    ctxdlydm1seqr(),
    ctxdlydm2seqr(),
    multidlydmseqr(),
    dmsseqr(),
    dnmsseqr(),
    dmcseqr(),
    dnmcseqr(),
    goseql(),
    rtgoseql(),
    dlygoseql(),
    antiseql(),
    rtantiseql(),
    dlyantiseql(),
    dm1seql(),
    dm2seql(),
    ctxdm1seql(),
    ctxdm2seql(),
    multidmseql(),
    dlydm1seql(),
    dlydm2seql(),
    ctxdlydm1seql(),
    ctxdlydm2seql(),
    multidlydmseql(),
    dmsseql(),
    dnmsseql(),
    dmcseql(),
    dnmcseql()]

    env_names = [
    "go",
    "rtgo",
    "dlygo",
    "anti",
    "rtanti",
    "dlyanti",
    "dm1",
    "dm2",
    "ctxdm1",
    "ctxdm2",
    "multidm",
    "dlydm1",
    "dlydm2",
    "ctxdlydm1",
    "ctxdlydm2",
    "multidlydm",
    "dms",
    "dnms",
    "dmc",
    "dnmc",
    "dlygointr",
    "dlygointl",
    "dlyantiintr",
    "dlyantiintl",
    "dlydm1intr",
    "dlydm1intl",
    "dlydm2intr",
    "dlydm2intl",
    "ctxdlydm1intr",
    "ctxdlydm1intl",
    "ctxdlydm2intr",
    "ctxdlydm2intl",
    "multidlydmintr",
    "multidlydmintl",
    "dmsintr",
    "dmsintl",
    "dnmsintr",
    "dnmsintl",
    "dmcintr",
    "dmcintl",
    "dnmcintr",
    "dnmcintl",
    "goseqr",
    "rtgoseqr",
    "dlygoseqr",
    "antiseqr",
    "rtantiseqr",
    "dlyantiseqr",
    "dm1seqr",
    "dm2seqr",
    "ctxdm1seqr",
    "ctxdm2seqr",
    "multidmseqr",
    "dlydm1seqr",
    "dlydm2seqr",
    "ctxdlydm1seqr",
    "ctxdlydm2seqr",
    "multidlydmseqr",
    "dmsseqr",
    "dnmsseqr",
    "dmcseqr",
    "dnmcseqr",
    "goseql",
    "rtgoseql",
    "dlygoseql",
    "antiseql",
    "rtantiseql",
    "dlyantiseql",
    "dm1seql",
    "dm2seql",
    "ctxdm1seql",
    "ctxdm2seql",
    "multidmseql",
    "dlydm1seql",
    "dlydm2seql",
    "ctxdlydm1seql",
    "ctxdlydm2seql",
    "multidlydmseql",
    "dmsseql",
    "dnmsseql",
    "dmcseql",
    "dnmcseql"]
    
    return envs, env_names