import { test } from 'node:test';
import assert from 'node:assert/strict';
import { acknowledgeDraft, answers, buildContext, changedFields, mergeList, parseHash, rebase, registerOpen, openList } from './src/copilot-context.js';

test('only the fields the user changed are reported, including manual clears', () => {
  assert.deepEqual(changedFields({ name: 'Old', notes: 'Before', mmproj: '/x.gguf' }, { name: 'Old', notes: 'Manual', mmproj: '' }), { notes: 'Manual', mmproj: '' });
  assert.deepEqual(changedFields({ a: 1 }, { a: 1 }), {});
});

test('a reload during an edit keeps the touched fields and takes the new value of the rest', () => {
  const oldBase = { name: 'Alfa', aliases: 'a', vision: false };
  const form = { name: 'Alfa 2', aliases: 'a', vision: false };          // the user typed a name
  const next = { name: 'Agent name', aliases: 'a\nb', vision: true };     // the assistant changed all three
  const r = rebase(oldBase, form, next);
  assert.deepEqual(r.form, { name: 'Alfa 2', aliases: 'a\nb', vision: true });
  assert.deepEqual(changedFields(r.base, r.form), { name: 'Alfa 2' });     // saving still sends only the name
});

test('a list edited by hand keeps what the assistant added meanwhile', () => {
  assert.deepEqual(mergeList(['a', 'b'], ['a', 'c'], ['a', 'b', 'x']), ['a', 'x', 'c']);
  assert.deepEqual(mergeList(['a'], ['a'], ['a', 'x']), ['a', 'x']);
});

test('hash routes are parsed with ids decoded', () => {
  assert.deepEqual(parseHash('#/ejecutar/r_1?x=1'), { page: 'ejecutar', param: 'r_1' });
  assert.deepEqual(parseHash(''), { page: '', param: null });
  assert.deepEqual(parseHash('#/pruebas/s%20a'), { page: 'pruebas', param: 's a' });
});

test('context-request answering: creation forms answer for no object, edit forms for theirs', () => {
  assert.equal(answers({ detail: { objectId: null } }, null), true);
  assert.equal(answers({ detail: { objectId: 'k_1' } }, null), false);
  assert.equal(answers({ detail: { objectId: 'k_1' } }, 'k_1'), true);
  assert.equal(answers({ detail: { objectId: null } }, 'k_1'), false);
});

const calls = [];
const call = async (name, args) => {
  calls.push([name, args]);
  if (name === 'suite_get' && args.suite === 's_mine') return { suite: { id: 's_mine', name: 'Mine', category: 'custom', builtin: false }, total_cases: 3 };
  if (name === 'case_get' && args.case === 'k_1') return { case: { id: 'k_1', title: 'Capital', suite_id: 's_mine', checker_label: 'contains', weight: 1 } };
  if (name === 'model_get' && args.model === 'c_1') return { model: { id: 'c_1', name: 'Alfa', enabled: true, kind: 'gguf', aliases: ['a'] } };
  if (name === 'run_status' && args.run === 'r_1234567890abc') return { run: { id: 'r_1234567890abc', state: 'running', suites: [{ id: 's', name: 'Quick' }], contestants: [{ id: 'c', name: 'Alfa' }] } };
  throw new Error('not found');
};

test('dashboard has no objects and never calls the app', async () => {
  calls.length = 0;
  const ctx = await buildContext({ hash: '#/', call });
  assert.deepEqual(ctx, { active_objects: [], view: { page: 'dashboard' } });
  assert.equal(calls.length, 0);
});

test('suite route plus the open case give real objects and the unsaved form', async () => {
  const off = registerOpen('case', 'k_1');
  const ctx = await buildContext({ hash: '#/pruebas/s_mine', call, open: openList(),
    editors: (id) => (id === 'k_1' ? [{ kind: 'case-form', dirty: true, fields: { notes: 'x' } }] : []) });
  off();
  assert.deepEqual(ctx.active_objects.map((o) => [o.kind, o.object_id, o.title]), [['suite', 's_mine', 'Mine'], ['case', 'k_1', 'Capital']]);
  assert.deepEqual(ctx.view, { page: 'tests', object_id: 'k_1', editor: { kind: 'case-form', dirty: true, fields: { notes: 'x' } } });
  assert.equal(openList().length, 0);
});

test('ids that do not exist yield no object and do not throw', async () => {
  const ctx = await buildContext({ hash: '#/ejecutar/r_missing', call, open: [{ kind: 'model', id: 'c_gone' }] });
  assert.deepEqual(ctx, { active_objects: [], view: { page: 'run' } });
});

test('run route and model drawer; creation forms are reported without an object', async () => {
  const ctx = await buildContext({ hash: '#/ejecutar/r_1234567890abc', call, open: [{ kind: 'model', id: 'c_1' }],
    editors: (id) => (id == null ? [{ kind: 'run-form', dirty: false, fields: {} }] : []) });
  assert.equal(ctx.active_objects[0].title, 'Run r_12345678 (running)');
  assert.deepEqual(ctx.active_objects[0].models, ['Alfa']);
  assert.equal(ctx.active_objects[1].object_id, 'c_1');
  assert.equal(ctx.view.editor.kind, 'run-form');
  const creation = await buildContext({ hash: '#/ejecutar', call, editors: () => [{ kind: 'run-form', dirty: true, fields: { suites: ['s'] } }] });
  assert.deepEqual(creation.active_objects, []);
  assert.equal(creation.view.editor.fields.suites[0], 's');
});

test('pending native context retains its form when navigation mounts another form', async () => {
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  let form = { kind: 'suite-form', dirty: true, fields: { description: 'Original draft' } };
  const snapshot = buildContext({ hash: '#/pruebas/s_mine', call: async (name, args) => { await pending; return call(name, args); },
    editors: (id) => id == null ? [form] : [] });
  form = { kind: 'settings-form', dirty: true, fields: { days: 7 } };
  release();
  const old = await snapshot;
  assert.equal(old.view.page, 'tests');
  assert.equal(old.active_objects[0].object_id, 's_mine');
  assert.equal(old.view.editor.kind, 'suite-form');
  assert.equal(old.view.editor.fields.description, 'Original draft');
  const next = await buildContext({ hash: '#/ajustes', call, editors: () => [form] });
  assert.equal(next.view.page, 'settings');
  assert.equal(next.view.editor.kind, 'settings-form');
});

test('acknowledging a settings save preserves later nested fields and sibling edits', () => {
  const sent = { 'routes.policy': { days: '7', top_k: '2' }, 'ui.language': 'en' };
  const pending = { 'routes.policy': { days: '8', top_k: '2', min_cases: '12' }, 'ui.language': 'en', 'runner.timeout_s': '150' };
  assert.deepEqual(acknowledgeDraft(pending, sent), { 'routes.policy': { days: '8', min_cases: '12' }, 'runner.timeout_s': '150' });
  assert.deepEqual(acknowledgeDraft(sent, sent), {});
});
