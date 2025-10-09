# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# 
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.

# gan_op = me.parent().op('gan_script')

def getSgModule():
    return me.parent().op('gan_script').module

def onValueChange(par, val, prev):
    if par.name == "Model":
        getSgModule().updateModel(val)
    else:
        getSgModule().updateBendingParameter(par.name, val)
    return

def onPulse(par):
    return

def onExpressionChange(par, val, prev):
    return

def onExportChange(par, val, prev):
    return

def onEnableChange(par, val, prev):
    return

def onModeChange(par, val, prev):
    return
    