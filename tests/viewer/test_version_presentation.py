import unittest
from types import SimpleNamespace
from utils.version_presentation import mapping,relabel,export_source_order,VersionPresentationRenderer,REVISION


class PresentationTests(unittest.TestCase):
    def test_source_preview_does_not_invent_experiment_versions(self):
        source=SimpleNamespace(info=dict(comparison=True,source_preview=True,name='新 data',
            left_label='原始文件',right_label='修正文件'))
        renderer=VersionPresentationRenderer(source);renderer.configure()
        self.assertEqual(renderer.info,source.info)
        self.assertEqual(renderer.orders,{})
        self.assertNotIn('version_presentation',renderer.info)

    def test_swap_is_simultaneous_and_preserves_other_versions(self):
        self.assertEqual(relabel('V1 / V2 / V3 / V4'),'V2 / V1 / V3 / V4')
        rows=mapping(['V1 · baseline','V2 · archived','V3 · final'])
        self.assertEqual([r['source_panel_index'] for r in rows],[1,0,2])
        self.assertEqual([r['display_label'] for r in rows],['V1 · archived','V2 · baseline','V3 · final'])

    def test_two_version_modes_keep_actual_assets(self):
        source=SimpleNamespace(info=dict(comparison=True,name='V1 / V3',left_label='V1 baseline',right_label='V3 final',
            mesh=dict(available=True,left_label='V1 mesh',right_label='V3 mesh',description='V1 / V3 mesh')))
        renderer=VersionPresentationRenderer(source);renderer.configure()
        self.assertEqual(renderer.orders['mesh'],[0,1])
        self.assertEqual(renderer.info['mode_version_labels']['mesh'],['V2 · mesh','V3 · mesh'])
        self.assertEqual(source.info['left_label'],'V1 baseline')

    def test_old_exports_are_mapped_and_new_exports_are_not_double_swapped(self):
        old=dict(versions=[dict(id=str(i),label=f'V{i+1}') for i in range(3)])
        self.assertEqual(export_source_order(old),('1','0','2'))
        current=dict(old,version_presentation=dict(revision=REVISION))
        self.assertEqual(export_source_order(current),('0','1','2'))

    def test_new_object_stages_have_explicit_order_and_own_name(self):
        source=SimpleNamespace(info=dict(comparison=True,name='Lamp · V1 / V2 / V3',
            version_labels=['V1 · 原始','V2 · 修正','V3 · 表面'],
            source_to_display={v:v for v in ('V1','V2','V3')}))
        renderer=VersionPresentationRenderer(source);renderer.configure()
        self.assertEqual(renderer.orders['gaussian'],[0,1,2])
        self.assertEqual(renderer.info['name'],'Lamp · V1 / V2 / V3')
        self.assertEqual(renderer.info['version_labels'][0],'V1 · 原始')
        self.assertNotIn('原V2',renderer.info['version_presentation']['disclosure'])
        manifest=dict(versions=[dict(id=f'v{i}',label=f'V{i}') for i in (1,2,3)],
                      version_presentation=renderer.info['version_presentation'])
        self.assertEqual(export_source_order(manifest),('v1','v2','v3'))

    def test_duplicate_stage_mapping_rejected(self):
        source=SimpleNamespace(info=dict(comparison=True,source_to_display={'V1':'V1','V2':'V1','V3':'V3'}))
        with self.assertRaises(ValueError):VersionPresentationRenderer(source).configure()


if __name__=='__main__':unittest.main()
