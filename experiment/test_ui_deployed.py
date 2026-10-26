import json
from pathlib import Path
import tempfile
import unittest
from .ui_deployed import freeze
from .protocol import sha256


class DeployedIdentityTests(unittest.TestCase):
    def test_cannot_replace_comparison_renderer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);catalog=root/'scenes.json'
            catalog.write_text(json.dumps(dict(scenes=[dict(id='lego',type='comparison')])) )
            with self.assertRaisesRegex(ValueError,'Retain the original renderer'):
                freeze(catalog,root)

    def test_source_hash_and_code_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'references').mkdir()
            (root/'references/reference.json').write_text('{}')
            source=root/'original.npz';source.write_bytes(b'frozen source identity')
            code=root/'viewer.py';code.write_text('# original renderer\n')
            item=dict(path=source.name,sha256=sha256(source))
            (root/'manifest.json').write_text(json.dumps(dict(format='gaussian_pair_v1',left=item,right=item)))
            catalog=root/'scenes.json';catalog.write_text(json.dumps(dict(scenes=[dict(
                id='metronome',type='gaussian_pair',path='manifest.json',reference_path='references/reference.json')])) )
            before=freeze(catalog,root)
            code.write_text('# changed renderer\n')
            self.assertNotEqual(before,freeze(catalog,root))
            source.write_bytes(b'replacement source')
            with self.assertRaisesRegex(ValueError,'asset hash mismatch'):
                freeze(catalog,root)


if __name__=='__main__':unittest.main()
