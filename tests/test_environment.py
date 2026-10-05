import sys
from types import SimpleNamespace
import pytest
from scripts import validate_environment as env


@pytest.mark.parametrize('peft_version',['0.21.1','0.21.2'])
def test_target_only_validation_imports_no_spec_packages(monkeypatch,tmp_path,peft_version):
    requirements=tmp_path/'requirements.txt'
    requirements.write_text('torch==2.13.0\npeft==0.21.1\n')
    monkeypatch.setattr(sys,'version_info',(3,12,3))
    monkeypatch.setattr(env,'version',lambda name: {'torch':'2.13.0+cu130','peft':peft_version}[name])
    imported=[]
    def fake_import(name):
        imported.append(name)
        if name=='torch':
            return SimpleNamespace(__version__='2.13.0+cu130',version=SimpleNamespace(cuda='13.0'),
                cuda=SimpleNamespace(is_available=lambda:True,get_device_capability=lambda:(10,0)))
        assert name=='peft'
        return SimpleNamespace(**{key:object() for key in ('LoraConfig','TaskType','get_peft_model',
            'get_peft_model_state_dict','set_peft_model_state_dict')})
    monkeypatch.setattr(env.importlib,'import_module',fake_import)
    env.main(['--requirements',str(requirements),'--require-cuda'])
    assert set(imported)=={'torch','peft'}


def test_environment_rejects_unvalidated_version(monkeypatch,tmp_path):
    requirements=tmp_path/'requirements.txt'; requirements.write_text('peft==0.21.1\n')
    monkeypatch.setattr(sys,'version_info',(3,12,3))
    monkeypatch.setattr(env,'version',lambda name:'0.0.0')
    with pytest.raises(RuntimeError,match='expected 0.21.1, found 0.0.0'):
        env.main(['--requirements',str(requirements)])
