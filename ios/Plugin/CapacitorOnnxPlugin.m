#import <Foundation/Foundation.h>
#import <Capacitor/Capacitor.h>

CAP_PLUGIN(CapacitorOnnxPlugin, "CapacitorOnnx",
    CAP_PLUGIN_METHOD(isActive, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(loadModel, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(run, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(release, CAPPluginReturnPromise);
)
