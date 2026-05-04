# Publish Guide

## Prerequisites

- npm account with publish permission for the @cantoo scope.
- npm authenticated locally.

Run:

npm login
npm whoami

## First publish

For a scoped package, publish with --access public.

pnpm build
pnpm typecheck
pnpm pack --dry-run
npm publish --access public

## Next releases

1. Bump the package version.
2. Rebuild and validate.
3. Publish.

npm version patch
pnpm build
pnpm typecheck
pnpm pack --dry-run
npm publish

Use npm version minor or npm version major when appropriate.

## Consumer update (Capacitor app)

After publishing a new version, app repositories should update and resync native platforms:

pnpm add @cantoo/capacitor-onnx@latest
pnpm cap sync android
