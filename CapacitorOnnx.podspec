require 'json'

package = JSON.parse(File.read(File.join(__dir__, 'package.json')))

Pod::Spec.new do |s|
  s.name = 'CapacitorOnnx'
  s.version = package['version']
  s.summary = package['description']
  s.license = package['license']
  s.homepage = 'https://github.com/cantoo-scribe/capacitor-onnx'
  s.author = package['author']
  s.source = { :git => 'https://github.com/cantoo-scribe/capacitor-onnx.git', :tag => s.version.to_s }
  s.source_files = 'ios/Plugin/**/*.{swift,h,m,c,cc,mm,cpp}'
  s.ios.deployment_target = '14.0'
  s.dependency 'Capacitor'
  s.dependency 'onnxruntime-objc', '~> 1.24'
  s.swift_version = '5.9'
end
