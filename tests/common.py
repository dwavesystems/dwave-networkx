# Copyright 2025 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


class NullError(BaseException):
    def __init__(self, *args):
        raise RuntimeError("do not use this exception class")


class GraphTesting:
    def assertGraphsEqual(self, G, H, msg=None):
        try:
            self.assertIsInstance(
                G, type(H), "graphs G and H should have the same type"
            )
            self.assertEqual(
                set(G.nodes),
                set(H.nodes),
                "graphs G and H should have the same node set",
            )
            self.assertEqual(
                len(G.edges),
                len(H.edges),
                "graphs G and H should have the same number of edges",
            )
            for e in G.edges:
                self.assertTrue(
                    H.has_edge(*e), "graphs G and H have different edge sets"
                )
        except NullError if msg is None else Exception as e:
            # Sorry for the weirdness here.  We don't want to create strings on
            # the happy path, so we define a NullError that is never raised so
            # exceptions fall straight through in that case.

            sub_msg, *_ = e.args
            self.fail(f"{msg} : {sub_msg}")
